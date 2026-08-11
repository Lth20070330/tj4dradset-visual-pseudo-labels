from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Calibration:
    """Calibration matrices using KITTI/TJ4DRadSet conventions."""

    p2: np.ndarray
    r0_rect: np.ndarray
    radar_to_camera: np.ndarray

    @property
    def camera_to_radar(self) -> np.ndarray:
        return np.linalg.inv(self.radar_to_camera)

    @classmethod
    def from_file(cls, path: str | Path) -> "Calibration":
        values: dict[str, np.ndarray] = {}
        for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            key, raw_values = raw_line.split(":", maxsplit=1)
            values[key] = np.fromstring(raw_values, sep=" ", dtype=np.float64)

        required = {"P2", "R0_rect", "Tr_velo_to_cam"}
        missing = required - values.keys()
        if missing:
            raise ValueError(f"Missing calibration fields: {sorted(missing)}")

        p2 = values["P2"].reshape(3, 4)
        r0 = np.eye(4, dtype=np.float64)
        r0[:3, :3] = values["R0_rect"].reshape(3, 3)
        radar_to_camera = np.eye(4, dtype=np.float64)
        radar_to_camera[:3, :] = values["Tr_velo_to_cam"].reshape(3, 4)
        return cls(p2=p2, r0_rect=r0, radar_to_camera=radar_to_camera)


@dataclass(frozen=True)
class KittiObject:
    category: str
    truncated: int
    occluded: int
    alpha: float
    bbox_2d: np.ndarray
    dimensions_hwl: np.ndarray
    location_camera: np.ndarray
    rotation_y: float

    @classmethod
    def from_line(cls, line: str) -> "KittiObject":
        fields = line.split()
        if len(fields) != 15:
            raise ValueError(f"Expected 15 KITTI fields, got {len(fields)}: {line!r}")
        return cls(
            category=fields[0],
            truncated=int(float(fields[1])),
            occluded=int(fields[2]),
            alpha=float(fields[3]),
            bbox_2d=np.asarray(fields[4:8], dtype=np.float64),
            dimensions_hwl=np.asarray(fields[8:11], dtype=np.float64),
            location_camera=np.asarray(fields[11:14], dtype=np.float64),
            rotation_y=float(fields[14]),
        )


class TJ4DRadSet:
    """Reader for the KITTI-like TJ4DRadSet directory layout."""

    RADAR_FEATURES = ("x", "y", "z", "radial_velocity", "range", "power", "alpha", "beta")

    def __init__(self, root: str | Path, split: str = "training") -> None:
        root = Path(root)
        self.root = root / split if (root / split).is_dir() else root
        self.radar_dir = self.root / "velodyne"
        self.calib_dir = self.root / "calib"
        self.label_dir = self.root / "label_2"
        self.image_dir = self.root / "image_2"
        for required in (self.radar_dir, self.calib_dir, self.label_dir):
            if not required.is_dir():
                raise FileNotFoundError(f"Required TJ4DRadSet directory not found: {required}")

    @property
    def frame_ids(self) -> list[str]:
        radar_ids = {p.stem for p in self.radar_dir.glob("*.bin")}
        calib_ids = {p.stem for p in self.calib_dir.glob("*.txt")}
        label_ids = {p.stem for p in self.label_dir.glob("*.txt")}
        return sorted(radar_ids & calib_ids & label_ids)

    def has_image(self, frame_id: str) -> bool:
        return any((self.image_dir / f"{frame_id}{suffix}").is_file() for suffix in (".png", ".jpg", ".jpeg"))

    def image_path(self, frame_id: str) -> Path:
        for suffix in (".png", ".jpg", ".jpeg"):
            path = self.image_dir / f"{frame_id}{suffix}"
            if path.is_file():
                return path
        raise FileNotFoundError(f"No image found for frame {frame_id} in {self.image_dir}")

    def load_image(self, frame_id: str) -> np.ndarray:
        """Load an RGB uint8 image."""
        with Image.open(self.image_path(frame_id)) as image:
            return np.asarray(image.convert("RGB"))

    def load_radar(self, frame_id: str) -> np.ndarray:
        path = self.radar_dir / f"{frame_id}.bin"
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % len(self.RADAR_FEATURES):
            raise ValueError(f"Radar file has {raw.size} floats, not divisible by 8: {path}")
        return raw.reshape(-1, len(self.RADAR_FEATURES))

    def load_calibration(self, frame_id: str) -> Calibration:
        return Calibration.from_file(self.calib_dir / f"{frame_id}.txt")

    def load_labels(self, frame_id: str) -> list[KittiObject]:
        path = self.label_dir / f"{frame_id}.txt"
        return [KittiObject.from_line(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

