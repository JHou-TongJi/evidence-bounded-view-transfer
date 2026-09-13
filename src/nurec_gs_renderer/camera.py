from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _matrix4(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} is not a homogeneous rigid transform")
    return matrix


def invert_pose(matrix: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform without a general matrix inverse."""
    matrix = _matrix4(matrix, "matrix")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -(rotation.T @ translation)
    return result


def camera_pose_to_opencv_w2c(
    matrix: np.ndarray,
    *,
    matrix_type: str = "camera_to_world",
    camera_axes: str = "opencv",
) -> np.ndarray:
    """Convert a camera pose to the OpenCV world-to-camera matrix gsplat expects.

    OpenCV camera axes are +x right, +y down, +z forward. OpenGL camera axes
    are +x right, +y up, -z forward.
    """
    matrix = _matrix4(matrix, "camera pose")
    if matrix_type not in {"camera_to_world", "world_to_camera"}:
        raise ValueError("matrix_type must be 'camera_to_world' or 'world_to_camera'")
    if camera_axes not in {"opencv", "opengl"}:
        raise ValueError("camera_axes must be 'opencv' or 'opengl'")

    c2w = matrix if matrix_type == "camera_to_world" else invert_pose(matrix)
    if camera_axes == "opengl":
        # Right-multiplication changes the camera basis without moving its origin.
        c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
    return invert_pose(c2w).astype(np.float32)


def compose_ncore_camera_pose(
    T_rig_world: np.ndarray,
    T_camera_rig: np.ndarray,
) -> np.ndarray:
    """Return T_camera_world (camera-to-world) using NCore conventions.

    NCore names T_source_target by the direction in which points transform.
    Therefore T_rig_world maps rig points into world coordinates and
    T_camera_rig maps camera points into the rig frame.
    """
    return _matrix4(T_rig_world, "T_rig_world") @ _matrix4(T_camera_rig, "T_camera_rig")


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def _quaternions_to_rotation_matrices_wxyz(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))


def interpolate_w2c(start: np.ndarray, end: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match gsplat's linear-translation/quaternion-slerp W2C interpolation."""
    start = _matrix4(start, "start world_to_camera")
    end = _matrix4(end, "end world_to_camera")
    q_start = _rotation_matrix_to_quaternion_wxyz(start[:3, :3])
    q_end = _rotation_matrix_to_quaternion_wxyz(end[:3, :3])
    dot = float(np.dot(q_start, q_end))
    if dot < 0.0:
        q_end = -q_end
        dot = -dot
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if dot > 0.9995:
        quaternions = (1.0 - times[:, None]) * q_start + times[:, None] * q_end
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        quaternions = (
            np.sin((1.0 - times) * theta)[:, None] / sin_theta * q_start
            + np.sin(times * theta)[:, None] / sin_theta * q_end
        )
    rotations = _quaternions_to_rotation_matrices_wxyz(quaternions)
    translations = (1.0 - times[:, None]) * start[:3, 3] + times[:, None] * end[:3, 3]
    return rotations, translations


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

    @property
    def K(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def resized(self, width: int | None = None, height: int | None = None) -> "CameraIntrinsics":
        """Resize output and scale the pinhole intrinsics consistently."""
        width = self.width if width is None else int(width)
        height = self.height if height is None else int(height)
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=width,
            height=height,
        )

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "CameraIntrinsics":
        return cls(
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            cx=float(value["cx"]),
            cy=float(value["cy"]),
            width=int(value["width"]),
            height=int(value["height"]),
        )


@dataclass(frozen=True)
class FThetaCameraParameters:
    """NCore/gsplat FTheta distortion and shutter parameters.

    The principal point and resolution remain in ``CameraIntrinsics`` because
    gsplat receives them through K and the rasterization width/height.
    """

    reference_poly: str
    pixeldist_to_angle_poly: tuple[float, ...]
    angle_to_pixeldist_poly: tuple[float, ...]
    max_angle: float
    linear_cde: tuple[float, ...]
    shutter_type: str = "GLOBAL"

    def __post_init__(self) -> None:
        if self.reference_poly not in {"PIXELDIST_TO_ANGLE", "ANGLE_TO_PIXELDIST"}:
            raise ValueError("unsupported FTheta reference polynomial")
        if len(self.pixeldist_to_angle_poly) != 6 or len(self.angle_to_pixeldist_poly) != 6:
            raise ValueError("FTheta polynomials must each contain six coefficients")
        if len(self.linear_cde) != 3:
            raise ValueError("FTheta linear_cde must contain three values")
        if self.max_angle <= 0.0:
            raise ValueError("FTheta max_angle must be positive")
        if self.shutter_type not in {
            "GLOBAL",
            "ROLLING_TOP_TO_BOTTOM",
            "ROLLING_LEFT_TO_RIGHT",
            "ROLLING_BOTTOM_TO_TOP",
            "ROLLING_RIGHT_TO_LEFT",
        }:
            raise ValueError(f"unsupported shutter type: {self.shutter_type}")
        values = (*self.pixeldist_to_angle_poly, *self.angle_to_pixeldist_poly, *self.linear_cde)
        if not np.all(np.isfinite(values)) or not np.isfinite(self.max_angle):
            raise ValueError("FTheta parameters contain non-finite values")

    @property
    def is_rolling(self) -> bool:
        return self.shutter_type != "GLOBAL"

    def resized(self, sx: float, sy: float) -> "FThetaCameraParameters":
        """Apply NCore's exact image-domain scaling to FTheta parameters."""
        if sx <= 0.0 or sy <= 0.0:
            raise ValueError("FTheta image scale must be positive")
        # Substitute pixel_distance / sy into the pixel-distance-to-angle
        # polynomial. The forward polynomial's result is measured in pixels.
        pixel_to_angle = tuple(value / (sy**degree) for degree, value in enumerate(self.pixeldist_to_angle_poly))
        angle_to_pixel = tuple(value * sy for value in self.angle_to_pixeldist_poly)
        c, d, e = self.linear_cde
        ratio = sx / sy
        return FThetaCameraParameters(
            reference_poly=self.reference_poly,
            pixeldist_to_angle_poly=pixel_to_angle,
            angle_to_pixeldist_poly=angle_to_pixel,
            max_angle=self.max_angle,
            linear_cde=(c * ratio, d * ratio, e),
            shutter_type=self.shutter_type,
        )


@dataclass(frozen=True)
class CameraFrame:
    frame_id: str
    intrinsics: CameraIntrinsics
    world_to_camera: np.ndarray
    timestamp_us: int | None = None
    camera_model: str = "pinhole"
    ftheta: FThetaCameraParameters | None = None
    world_to_camera_end: np.ndarray | None = None
    timestamp_start_us: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "world_to_camera", _matrix4(self.world_to_camera, "world_to_camera").astype(np.float32))
        if self.world_to_camera_end is not None:
            object.__setattr__(
                self,
                "world_to_camera_end",
                _matrix4(self.world_to_camera_end, "world_to_camera_end").astype(np.float32),
            )
        if self.camera_model not in {"pinhole", "ftheta"}:
            raise ValueError("camera_model must be 'pinhole' or 'ftheta'")
        if self.camera_model == "ftheta" and self.ftheta is None:
            raise ValueError("ftheta camera requires FTheta parameters")
        if self.camera_model == "pinhole" and self.ftheta is not None:
            raise ValueError("pinhole camera cannot contain FTheta parameters")
        if self.ftheta is not None and self.ftheta.is_rolling and self.world_to_camera_end is None:
            raise ValueError("rolling-shutter camera requires an end-of-exposure pose")


@dataclass(frozen=True)
class CameraPath:
    frames: tuple[CameraFrame, ...]
    source: str
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("camera path must contain at least one frame")

    @classmethod
    def from_frames(
        cls,
        frames: Iterable[CameraFrame],
        source: str,
        metadata: dict[str, object] | None = None,
    ) -> "CameraPath":
        return cls(tuple(frames), source, metadata)

    def resized(self, width: int | None = None, height: int | None = None) -> "CameraPath":
        """Return the same poses with consistently resized output intrinsics."""
        def resize_frame(frame: CameraFrame) -> CameraFrame:
            width_out = frame.intrinsics.width if width is None else int(width)
            height_out = frame.intrinsics.height if height is None else int(height)
            if frame.ftheta is None:
                intrinsics = frame.intrinsics.resized(width_out, height_out)
                ftheta = None
            else:
                sx = width_out / frame.intrinsics.width
                sy = height_out / frame.intrinsics.height
                # NCore FTheta parameters use the centre of the first pixel as
                # (0, 0), hence the half-pixel-aware principal-point scaling.
                intrinsics = CameraIntrinsics(
                    fx=frame.intrinsics.fx * sx,
                    fy=frame.intrinsics.fy * sy,
                    cx=(frame.intrinsics.cx + 0.5) * sx - 0.5,
                    cy=(frame.intrinsics.cy + 0.5) * sy - 0.5,
                    width=width_out,
                    height=height_out,
                )
                ftheta = frame.ftheta.resized(sx, sy)
            return CameraFrame(
                frame_id=frame.frame_id,
                intrinsics=intrinsics,
                world_to_camera=frame.world_to_camera,
                timestamp_us=frame.timestamp_us,
                camera_model=frame.camera_model,
                ftheta=ftheta,
                world_to_camera_end=frame.world_to_camera_end,
                timestamp_start_us=frame.timestamp_start_us,
            )

        return CameraPath.from_frames(
            (resize_frame(frame) for frame in self.frames),
            source=self.source,
            metadata=self.metadata,
        )
