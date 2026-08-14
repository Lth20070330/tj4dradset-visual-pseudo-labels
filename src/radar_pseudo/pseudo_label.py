from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .dataset import Calibration
from .geometry import project_camera_points, radar_to_rectified_camera
from .vision import Instance2D


CLASS_DIMENSIONS_HWL = {
    "Car": np.array([1.55, 1.80, 4.30], dtype=np.float64),
    "Truck": np.array([3.00, 2.50, 8.00], dtype=np.float64),
    "Pedestrian": np.array([1.70, 0.60, 0.60], dtype=np.float64),
    "Cyclist": np.array([1.70, 0.80, 1.80], dtype=np.float64),
    "Other": np.array([1.80, 1.80, 4.00], dtype=np.float64),
}

CLASS_DIMENSION_LIMITS_HWL = {
    "Car": (np.array([1.2, 1.5, 3.2]), np.array([2.1, 2.3, 5.8])),
    "Truck": (np.array([2.0, 1.8, 4.5]), np.array([4.2, 3.2, 14.0])),
    "Pedestrian": (np.array([1.3, 0.35, 0.35]), np.array([2.2, 1.0, 1.2])),
    "Cyclist": (np.array([1.3, 0.45, 1.0]), np.array([2.2, 1.2, 2.8])),
    "Other": (np.array([1.0, 0.8, 1.0]), np.array([4.0, 3.5, 10.0])),
}


@dataclass(frozen=True)
class PseudoBox3D:
    category: str
    dimensions_hwl: np.ndarray
    location_camera: np.ndarray
    rotation_y: float
    visual_confidence: float
    visual_depth_m: float
    radar_depth_m: float | None
    radar_points: int
    quality: float
    position_source: str
    bbox_2d: np.ndarray
    class_quality: float = 1.0
    center_quality: float = 1.0
    size_quality: float = 1.0
    yaw_quality: float = 1.0
    cluster_candidates: int = 0
    cluster_score: float = 0.0
    candidate_margin: float = 0.0
    geometry_confidence: float = 0.0

    def to_dict(self) -> dict[str, object]:
        record = asdict(self)
        for key in ("dimensions_hwl", "location_camera", "bbox_2d"):
            record[key] = record[key].tolist()
        return record


def backproject_pixel(u: float, v: float, depth: float, p2: np.ndarray) -> np.ndarray:
    fx, fy = p2[0, 0], p2[1, 1]
    cx, cy = p2[0, 2], p2[1, 2]
    return np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], dtype=np.float64)


def densest_depth_cluster(depths: np.ndarray, gap_m: float = 2.0) -> np.ndarray:
    depths = np.sort(np.asarray(depths, dtype=np.float64))
    if len(depths) == 0:
        return depths
    split_indices = np.flatnonzero(np.diff(depths) > gap_m) + 1
    clusters = np.split(depths, split_indices)
    return max(clusters, key=lambda values: (len(values), -np.median(values)))


@dataclass(frozen=True)
class RadarClusterCandidate:
    camera_points: np.ndarray
    radar_points: np.ndarray
    score: float
    depth_agreement: float
    support: float
    compactness: float
    velocity_coherence: float
    mask_centrality: float


def _candidate_radar_indices(
    radar: np.ndarray,
    calibration: Calibration,
    instance: Instance2D,
    visual_depth_m: float,
    relative_gate: float = 0.50,
    minimum_gate_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = radar_to_rectified_camera(radar[:, :3], calibration)
    pixels, depth = project_camera_points(camera, calibration.p2)
    u = np.rint(pixels[:, 0]).astype(int)
    v = np.rint(pixels[:, 1]).astype(int)
    height, width = instance.mask.shape
    valid = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    in_mask = np.zeros(len(radar), dtype=bool)
    in_mask[valid] = instance.mask[v[valid], u[valid]]
    gate = max(minimum_gate_m, relative_gate * visual_depth_m)
    chosen = np.flatnonzero(in_mask & (np.abs(depth - visual_depth_m) <= gate))
    return camera, pixels, depth, chosen


def radar_cluster_candidates(
    radar: np.ndarray,
    calibration: Calibration,
    instance: Instance2D,
    visual_depth_m: float,
    gap_m: float = 2.0,
) -> list[RadarClusterCandidate]:
    camera, pixels, depth, chosen = _candidate_radar_indices(
        radar, calibration, instance, visual_depth_m
    )
    if not len(chosen):
        return []
    ordered = chosen[np.argsort(depth[chosen])]
    boundaries = np.flatnonzero(np.diff(depth[ordered]) > gap_m) + 1
    clusters = np.split(ordered, boundaries)
    x1, y1, x2, y2 = instance.bbox_xyxy
    box_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)
    box_scale = max(float(np.hypot(x2 - x1, y2 - y1)) * 0.5, 1.0)
    candidates: list[RadarClusterCandidate] = []
    for indices in clusters:
        if not len(indices):
            continue
        cluster_depths = depth[indices]
        median_depth = float(np.median(cluster_depths))
        depth_agreement = float(np.exp(-abs(median_depth - visual_depth_m) / max(8.0, 0.18 * visual_depth_m)))
        support = float(1.0 - np.exp(-len(indices) / 4.0))
        median_absolute_deviation = float(np.median(np.abs(cluster_depths - median_depth)))
        compactness = float(np.exp(-median_absolute_deviation / 1.5))
        velocity_coherence = float(np.exp(-np.std(radar[indices, 3]) / 3.0)) if len(indices) > 1 else 0.55
        normalized_offset = np.linalg.norm(np.median(pixels[indices], axis=0) - box_center) / box_scale
        mask_centrality = float(np.exp(-normalized_offset))
        score = float(
            0.30 * depth_agreement
            + 0.25 * support
            + 0.20 * compactness
            + 0.10 * velocity_coherence
            + 0.15 * mask_centrality
        )
        candidates.append(
            RadarClusterCandidate(
                camera_points=camera[indices],
                radar_points=radar[indices],
                score=score,
                depth_agreement=depth_agreement,
                support=support,
                compactness=compactness,
                velocity_coherence=velocity_coherence,
                mask_centrality=mask_centrality,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def _angle_distance_mod_pi(left: float, right: float) -> float:
    difference = abs(left - right) % np.pi
    return min(difference, np.pi - difference)


def fit_adaptive_geometry(
    category: str,
    camera_points: np.ndarray,
    location_camera: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Fit size/yaw with class-prior shrinkage; return dimensions, yaw, confidence."""
    prior = CLASS_DIMENSIONS_HWL[category].copy()
    if len(camera_points) < 3:
        return prior, -np.pi / 2, 0.0
    horizontal = camera_points[:, [0, 2]]
    centered = horizontal - np.median(horizontal, axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    contrast = float((eigenvalues[0] - eigenvalues[1]) / max(eigenvalues.sum(), 1e-6))
    spread = float(np.sqrt(max(eigenvalues[0], 0.0)))
    support_confidence = float(1.0 - np.exp(-len(camera_points) / 7.0))
    geometry_confidence = float(np.clip(contrast * support_confidence * min(1.0, spread / 0.8), 0.0, 1.0))

    principal = eigenvectors[:, 0]
    principal_yaw = float(np.arctan2(-principal[1], principal[0]))
    perpendicular_yaw = principal_yaw + np.pi / 2
    viewing_yaw = float(np.arctan2(-location_camera[2], location_camera[0]))
    yaw = min(
        (principal_yaw, perpendicular_yaw),
        key=lambda candidate: _angle_distance_mod_pi(candidate, viewing_yaw),
    )
    yaw = float((yaw + np.pi) % (2 * np.pi) - np.pi)

    cosine, sine = np.cos(yaw), np.sin(yaw)
    length_axis = np.array([cosine, -sine], dtype=np.float64)
    width_axis = np.array([sine, cosine], dtype=np.float64)
    length_extent = float(np.ptp(centered @ length_axis))
    width_extent = float(np.ptp(centered @ width_axis))
    height_extent = float(np.ptp(camera_points[:, 1]))
    observed = np.array(
        [height_extent + 0.35, width_extent + 0.45, length_extent + 0.80], dtype=np.float64
    )
    shrinkage = min(0.55, geometry_confidence)
    dimensions = (1.0 - shrinkage) * prior + shrinkage * observed
    lower, upper = CLASS_DIMENSION_LIMITS_HWL[category]
    dimensions = np.clip(dimensions, lower, upper)
    return dimensions, yaw, geometry_confidence


def radar_points_for_instance(
    radar: np.ndarray,
    calibration: Calibration,
    instance: Instance2D,
    visual_depth_m: float,
    relative_gate: float = 0.35,
    minimum_gate_m: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    camera = radar_to_rectified_camera(radar[:, :3], calibration)
    pixels, depth = project_camera_points(camera, calibration.p2)
    u = np.rint(pixels[:, 0]).astype(int)
    v = np.rint(pixels[:, 1]).astype(int)
    height, width = instance.mask.shape
    valid = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    in_mask = np.zeros(len(radar), dtype=bool)
    in_mask[valid] = instance.mask[v[valid], u[valid]]
    gate = max(minimum_gate_m, relative_gate * visual_depth_m)
    candidates = in_mask & (np.abs(depth - visual_depth_m) <= gate)
    if not candidates.any():
        return camera[:0], radar[:0]
    candidate_indices = np.flatnonzero(candidates)
    cluster = densest_depth_cluster(depth[candidates])
    low, high = cluster.min() - 1e-6, cluster.max() + 1e-6
    chosen = candidate_indices[(depth[candidate_indices] >= low) & (depth[candidate_indices] <= high)]
    return camera[chosen], radar[chosen]


def estimate_pseudo_box(
    instance: Instance2D,
    depth_map_m: np.ndarray,
    radar: np.ndarray,
    calibration: Calibration,
) -> PseudoBox3D:
    mask_depths = depth_map_m[instance.mask]
    valid_depths = mask_depths[np.isfinite(mask_depths) & (mask_depths > 0)]
    if len(valid_depths) < 10:
        raise ValueError("Instance mask does not contain enough valid depth pixels")
    visual_depth = float(np.median(valid_depths))
    camera_hits, _ = radar_points_for_instance(radar, calibration, instance, visual_depth)
    dimensions = CLASS_DIMENSIONS_HWL[instance.target_class or "Other"].copy()
    x1, _, x2, y2 = instance.bbox_xyxy
    u_center = (x1 + x2) / 2

    if len(camera_hits):
        radar_depth = float(np.median(camera_hits[:, 2]))
        center_depth = radar_depth + 0.30 * dimensions[2]
        source = "radar_corrected"
    else:
        radar_depth = None
        center_depth = visual_depth
        source = "visual_depth"
    location = backproject_pixel(u_center, y2, center_depth, calibration.p2)

    depth_agreement = 0.0 if radar_depth is None else float(np.exp(-abs(visual_depth - radar_depth) / 12.0))
    radar_support = min(1.0, len(camera_hits) / 5.0)
    quality = float(np.clip(0.55 * instance.confidence + 0.25 * depth_agreement + 0.20 * radar_support, 0, 1))
    return PseudoBox3D(
        category=instance.target_class or "Other",
        dimensions_hwl=dimensions,
        location_camera=location,
        rotation_y=-np.pi / 2,
        visual_confidence=instance.confidence,
        visual_depth_m=visual_depth,
        radar_depth_m=radar_depth,
        radar_points=len(camera_hits),
        quality=quality,
        position_source=source,
        bbox_2d=instance.bbox_xyxy.copy(),
    )


def estimate_pseudo_box_mgu(
    instance: Instance2D,
    depth_map_m: np.ndarray,
    radar: np.ndarray,
    calibration: Calibration,
    adaptive_geometry: bool = False,
) -> PseudoBox3D:
    """MGU-PL: multi-hypothesis association and attribute-wise uncertainty."""
    mask_depths = depth_map_m[instance.mask]
    valid_depths = mask_depths[np.isfinite(mask_depths) & (mask_depths > 0)]
    if len(valid_depths) < 10:
        raise ValueError("Instance mask does not contain enough valid depth pixels")
    visual_depth = float(np.median(valid_depths))
    category = instance.target_class or "Other"
    prior_dimensions = CLASS_DIMENSIONS_HWL[category].copy()
    candidates = radar_cluster_candidates(radar, calibration, instance, visual_depth)
    best = candidates[0] if candidates else None
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    candidate_margin = (best.score - second_score) if best else 0.0
    x1, _, x2, y2 = instance.bbox_xyxy
    u_center = (x1 + x2) * 0.5

    if best is not None:
        radar_depth = float(np.median(best.camera_points[:, 2]))
        center_depth = radar_depth + 0.25 * prior_dimensions[2]
        location = backproject_pixel(u_center, y2, center_depth, calibration.p2)
        if adaptive_geometry:
            dimensions, rotation_y, geometry_confidence = fit_adaptive_geometry(
                category, best.camera_points, location
            )
            source = "mgu_radar_pca_geometry"
        else:
            # Sparse single-frame radar points do not support stable PCA box fitting.
            # Keep the robust class prior here; optional, class-gated reprojection is
            # performed as an explicit second stage.
            dimensions = prior_dimensions
            rotation_y = -np.pi / 2
            geometry_confidence = 0.0
            source = "mgu_radar_prior"
        radar_points = len(best.camera_points)
        ambiguity_confidence = float(np.clip(candidate_margin / 0.20, 0.0, 1.0))
        center_quality = float(
            np.clip(0.45 * best.score + 0.25 * best.support + 0.15 * best.compactness + 0.15 * ambiguity_confidence, 0, 1)
        )
        if adaptive_geometry:
            size_quality = float(np.clip(center_quality * (0.35 + 0.65 * geometry_confidence), 0, 1))
            yaw_quality = float(np.clip(center_quality * geometry_confidence, 0, 1))
        else:
            size_quality = float(0.55 * center_quality)
            yaw_quality = float(0.25 * center_quality)
        cluster_score = best.score
    else:
        radar_depth = None
        location = backproject_pixel(u_center, y2, visual_depth, calibration.p2)
        dimensions = prior_dimensions
        rotation_y = -np.pi / 2
        geometry_confidence = 0.0
        source = "mgu_visual_only"
        radar_points = 0
        center_quality = float(0.20 * instance.confidence)
        size_quality = float(0.10 * instance.confidence)
        yaw_quality = 0.0
        cluster_score = 0.0

    class_quality = float(instance.confidence)
    quality = float(
        np.clip(0.40 * class_quality + 0.40 * center_quality + 0.10 * size_quality + 0.10 * yaw_quality, 0, 1)
    )
    return PseudoBox3D(
        category=category,
        dimensions_hwl=dimensions,
        location_camera=location,
        rotation_y=rotation_y,
        visual_confidence=instance.confidence,
        visual_depth_m=visual_depth,
        radar_depth_m=radar_depth,
        radar_points=radar_points,
        quality=quality,
        position_source=source,
        bbox_2d=instance.bbox_xyxy.copy(),
        class_quality=class_quality,
        center_quality=center_quality,
        size_quality=size_quality,
        yaw_quality=yaw_quality,
        cluster_candidates=len(candidates),
        cluster_score=cluster_score,
        candidate_margin=candidate_margin,
        geometry_confidence=geometry_confidence,
    )


def estimate_pseudo_box_by_method(
    method: str,
    instance: Instance2D,
    depth_map_m: np.ndarray,
    radar: np.ndarray,
    calibration: Calibration,
) -> PseudoBox3D:
    if method == "b0":
        return estimate_pseudo_box(instance, depth_map_m, radar, calibration)
    if method == "mgu":
        return estimate_pseudo_box_mgu(instance, depth_map_m, radar, calibration)
    if method == "mgu_pca":
        return estimate_pseudo_box_mgu(instance, depth_map_m, radar, calibration, adaptive_geometry=True)
    raise ValueError(f"Unknown pseudo-label method: {method}")
