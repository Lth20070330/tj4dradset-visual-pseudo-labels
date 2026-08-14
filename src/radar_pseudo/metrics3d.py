from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import KittiObject


@dataclass(frozen=True)
class ScoredObject:
    frame_id: str
    obj: KittiObject
    score: float = 1.0


def polygon_area(polygon: np.ndarray) -> float:
    polygon = np.asarray(polygon, dtype=np.float64)
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _signed_polygon_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    edge = edge_end - edge_start
    relative = point - edge_start
    return float(edge[0] * relative[1] - edge[1] * relative[0]) >= -1e-10


def _line_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray:
    first = first_end - first_start
    second = second_end - second_start
    denominator = first[0] * second[1] - first[1] * second[0]
    if abs(denominator) < 1e-12:
        return (first_end + second_start) * 0.5
    delta = second_start - first_start
    fraction = (delta[0] * second[1] - delta[1] * second[0]) / denominator
    return first_start + fraction * first


def convex_polygon_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Clip one convex polygon against another using Sutherland-Hodgman."""
    output = np.asarray(subject, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)
    if len(output) < 3 or len(clip) < 3:
        return np.empty((0, 2), dtype=np.float64)
    if _signed_polygon_area(output) < 0:
        output = output[::-1]
    if _signed_polygon_area(clip) < 0:
        clip = clip[::-1]
    for edge_index in range(len(clip)):
        edge_start = clip[edge_index]
        edge_end = clip[(edge_index + 1) % len(clip)]
        input_polygon = output
        if len(input_polygon) == 0:
            break
        clipped: list[np.ndarray] = []
        previous = input_polygon[-1]
        previous_inside = _inside(previous, edge_start, edge_end)
        for current in input_polygon:
            current_inside = _inside(current, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    clipped.append(_line_intersection(previous, current, edge_start, edge_end))
                clipped.append(current)
            elif previous_inside:
                clipped.append(_line_intersection(previous, current, edge_start, edge_end))
            previous, previous_inside = current, current_inside
        output = np.asarray(clipped, dtype=np.float64)
    return output.reshape(-1, 2)


def bev_corners_camera(obj: KittiObject) -> np.ndarray:
    """Return the four bottom-plane corners in camera x-z coordinates."""
    _, width, length = obj.dimensions_hwl
    local = np.array(
        [[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]],
        dtype=np.float64,
    )
    cosine, sine = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    # KITTI yaw maps local (x, z) to (c*x+s*z, -s*x+c*z).
    kitti_rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return local @ kitti_rotation.T + obj.location_camera[[0, 2]]


def bev_iou(left: KittiObject, right: KittiObject) -> float:
    left_polygon = bev_corners_camera(left)
    right_polygon = bev_corners_camera(right)
    intersection = polygon_area(convex_polygon_intersection(left_polygon, right_polygon))
    union = polygon_area(left_polygon) + polygon_area(right_polygon) - intersection
    return intersection / union if union > 0 else 0.0


def iou_3d(left: KittiObject, right: KittiObject) -> float:
    intersection_area = polygon_area(convex_polygon_intersection(bev_corners_camera(left), bev_corners_camera(right)))
    left_top, left_bottom = left.location_camera[1] - left.dimensions_hwl[0], left.location_camera[1]
    right_top, right_bottom = right.location_camera[1] - right.dimensions_hwl[0], right.location_camera[1]
    intersection_height = max(0.0, min(left_bottom, right_bottom) - max(left_top, right_top))
    intersection = intersection_area * intersection_height
    left_volume = float(np.prod(left.dimensions_hwl))
    right_volume = float(np.prod(right.dimensions_hwl))
    union = left_volume + right_volume - intersection
    return intersection / union if union > 0 else 0.0


def center_distance_bev(left: KittiObject, right: KittiObject) -> float:
    return float(np.linalg.norm(left.location_camera[[0, 2]] - right.location_camera[[0, 2]]))


def orientation_error(left: KittiObject, right: KittiObject) -> float:
    difference = abs(float(left.rotation_y - right.rotation_y)) % (2 * np.pi)
    return min(difference, 2 * np.pi - difference)


def aligned_scale_error(left: KittiObject, right: KittiObject) -> float:
    intersection = float(np.prod(np.minimum(left.dimensions_hwl, right.dimensions_hwl)))
    union = float(np.prod(left.dimensions_hwl) + np.prod(right.dimensions_hwl) - intersection)
    return 1.0 - intersection / union if union > 0 else 1.0


def ap_r40(recall: np.ndarray, precision: np.ndarray) -> float:
    """Forty-point interpolated AP, reported as a fraction in [0, 1]."""
    if len(recall) == 0:
        return 0.0
    levels = np.linspace(0.0, 1.0, 41, dtype=np.float64)[1:]
    interpolated = [float(np.max(precision[recall >= level])) if np.any(recall >= level) else 0.0 for level in levels]
    return float(np.mean(interpolated))


def evaluate_ranked_detections(
    predictions: list[ScoredObject],
    ground_truth: dict[str, list[KittiObject]],
    overlap_threshold: float,
    overlap_function,
) -> dict[str, object]:
    total_ground_truth = sum(len(objects) for objects in ground_truth.values())
    matched = {frame_id: np.zeros(len(objects), dtype=bool) for frame_id, objects in ground_truth.items()}
    true_positive = np.zeros(len(predictions), dtype=np.float64)
    false_positive = np.zeros(len(predictions), dtype=np.float64)
    for prediction_index, prediction in enumerate(sorted(predictions, key=lambda item: item.score, reverse=True)):
        candidates = ground_truth.get(prediction.frame_id, [])
        if not candidates:
            false_positive[prediction_index] = 1.0
            continue
        overlaps = np.asarray([overlap_function(prediction.obj, candidate) for candidate in candidates], dtype=np.float64)
        overlaps[matched[prediction.frame_id]] = -1.0
        best_index = int(np.argmax(overlaps))
        if overlaps[best_index] >= overlap_threshold:
            true_positive[prediction_index] = 1.0
            matched[prediction.frame_id][best_index] = True
        else:
            false_positive[prediction_index] = 1.0
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / max(total_ground_truth, 1)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    return {
        "ground_truth": total_ground_truth,
        "predictions": len(predictions),
        "true_positives": int(true_positive.sum()),
        "precision": float(precision[-1]) if len(precision) else 0.0,
        "recall": float(recall[-1]) if len(recall) else 0.0,
        "ap_r40": ap_r40(recall, precision),
    }
