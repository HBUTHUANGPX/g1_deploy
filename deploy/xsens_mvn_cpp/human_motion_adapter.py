from __future__ import annotations

from typing import Sequence

import numpy as np

from deploy.utils.math_func import quat_mul
from deploy.xsens_mvn_cpp.types import (
    HUMAN_JOINT_TO_XSENS_JOINT_SEGMENTS,
    HumanMotionSample,
    XSENS_TO_HUMAN_JOINT,
)


class XsensRawFrameHumanMotionAdapter:
    """Convert one pybind Xsens raw frame into deploy human-motion arrays.

    Preconditions:
    - The input frame exposes a `segments` iterable.
    - Each segment exposes `name`, `position`, and `orientation` attributes.

    Postconditions:
    - Returned arrays are ordered by `desired_joint_names`.
    - Missing joints are filled with configured default values and marked invalid.
    - No coordinate conversion or joint-angle-to-quaternion conversion is performed.
    """

    def __init__(
        self,
        desired_joint_names: Sequence[str],
        name_map: dict[str, str] | None = None,
        missing_position: Sequence[float] = (0.0, 0.0, 0.0),
        missing_quaternion_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ):
        """Initialize the adapter with a stable desired human joint order.

        Preconditions:
        - `desired_joint_names` contains the consumer's expected human joint names.

        Postconditions:
        - The adapter is ready to map raw Xsens segment names into that order.
        """

        self.desired_joint_names = list(desired_joint_names)
        self.name_map = dict(XSENS_TO_HUMAN_JOINT)
        if name_map is not None:
            self.name_map.update(name_map)
        self.joint_angle_map = dict(HUMAN_JOINT_TO_XSENS_JOINT_SEGMENTS)
        self.missing_position = np.asarray(missing_position, dtype=np.float32)
        self.missing_quaternion_wxyz = np.asarray(
            missing_quaternion_wxyz, dtype=np.float32
        )

    def to_human_sample(self, frame) -> HumanMotionSample:
        """Convert one latest raw frame into a `HumanMotionSample`.

        Preconditions:
        - `frame` is a snapshot returned by `xsens_mvn_cpp_py`.

        Postconditions:
        - Segment world positions, segment world quaternions, and streamed
          joint angles are copied to numpy arrays.
        """

        segments_by_name = {segment.name: segment for segment in frame.segments}
        positions = np.zeros((len(self.desired_joint_names), 3), dtype=np.float32)
        quaternions = np.zeros((len(self.desired_joint_names), 4), dtype=np.float32)
        valid_mask = np.zeros(len(self.desired_joint_names), dtype=bool)
        joint_angles, joint_angle_valid_mask = self._map_joint_angles(frame)

        for index, joint_name in enumerate(self.desired_joint_names):
            xsens_name = self.name_map.get(joint_name, joint_name)
            segment = segments_by_name.get(xsens_name)
            if segment is None:
                positions[index] = self.missing_position
                quaternions[index] = self.missing_quaternion_wxyz
                continue

            positions[index] = self._vector3_to_array(segment.position)
            quaternions[index] = self._quaternion_to_array(segment.orientation)
            valid_mask[index] = True

        joint_quaternions = self._joint_angles_to_quaternions(
            joint_angles,
            joint_angle_valid_mask,
        )

        frame_time = int(getattr(frame, "frame_time", 0))
        sequence = int(getattr(frame, "sequence", 0))
        sample_counter = int(getattr(frame, "sample_counter", 0))
        return HumanMotionSample(
            joint_names=list(self.desired_joint_names),
            human_body_pos_w=positions,
            human_body_quat_w=quaternions,
            human_joint_quat=joint_quaternions,
            human_joint_angles=joint_angles,
            valid_mask=valid_mask,
            joint_angle_valid_mask=joint_angle_valid_mask,
            timestamp_ns=frame_time * 1_000_000,
            frame_id=f"xsens_raw:{sequence}",
            source_sample_counter=sample_counter,
            source_datagram_sequence=sequence,
        )

    def _map_joint_angles(self, frame) -> tuple[np.ndarray, np.ndarray]:
        """Map streamed MVN joint-angle entries into desired human joint order.

        Preconditions:
        - `frame.joints` contains raw type-20 entries parsed from MVN.

        Postconditions:
        - Returns raw angle triplets in stream units without conversion.
        - Missing entries are zero-filled and marked invalid.
        """

        angles = np.zeros((len(self.desired_joint_names), 3), dtype=np.float32)
        valid_mask = np.zeros(len(self.desired_joint_names), dtype=bool)
        joints_by_pair = {
            (joint.parent_segment_id, joint.child_segment_id): joint
            for joint in getattr(frame, "joints", [])
        }

        for index, joint_name in enumerate(self.desired_joint_names):
            pair = self.joint_angle_map.get(joint_name)
            if pair is None:
                continue
            joint = joints_by_pair.get(pair)
            if joint is None:
                continue
            angles[index] = self._vector3_to_array(joint.angles)
            valid_mask[index] = True
        return angles, valid_mask

    @staticmethod
    def _vector3_to_array(vector) -> np.ndarray:
        """Return a float32 array from an Xsens vector object.

        Preconditions:
        - `vector` exposes `x`, `y`, and `z` attributes.

        Postconditions:
        - A shape `(3,)` float32 array is returned.
        """

        return np.asarray([vector.x, vector.y, vector.z], dtype=np.float32)

    @staticmethod
    def _quaternion_to_array(quaternion) -> np.ndarray:
        """Return a float32 wxyz quaternion array from an Xsens quaternion object.

        Preconditions:
        - `quaternion` exposes `w`, `x`, `y`, and `z` attributes.

        Postconditions:
        - A shape `(4,)` float32 array is returned without normalization.
        """

        return np.asarray(
            [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
            dtype=np.float32,
        )

    def _joint_angles_to_quaternions(
        self,
        joint_angles_xyz: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Convert streamed MVN joint-angle triplets to local quaternions.

        Preconditions:
        - `joint_angles_xyz` is ordered by `desired_joint_names`.
        - Angles are MVN type-20 x/y/z values in degrees.

        Postconditions:
        - Valid entries are converted using an xyz rotation composition.
        - Missing entries use the configured identity/default quaternion.
        """

        joint_quat = np.tile(
            self.missing_quaternion_wxyz,
            (len(self.desired_joint_names), 1),
        ).astype(np.float32)
        if not valid_mask.any():
            return joint_quat

        valid_angles = np.deg2rad(joint_angles_xyz[valid_mask].astype(np.float32))
        half_angles = valid_angles * 0.5
        cos_half = np.cos(half_angles)
        sin_half = np.sin(half_angles)

        qx = np.stack(
            (
                cos_half[:, 0],
                sin_half[:, 0],
                np.zeros_like(sin_half[:, 0]),
                np.zeros_like(sin_half[:, 0]),
            ),
            axis=-1,
        )
        qy = np.stack(
            (
                cos_half[:, 1],
                np.zeros_like(sin_half[:, 1]),
                sin_half[:, 1],
                np.zeros_like(sin_half[:, 1]),
            ),
            axis=-1,
        )
        qz = np.stack(
            (
                cos_half[:, 2],
                np.zeros_like(sin_half[:, 2]),
                np.zeros_like(sin_half[:, 2]),
                sin_half[:, 2],
            ),
            axis=-1,
        )
        joint_quat[valid_mask] = quat_mul(quat_mul(qx, qy), qz).astype(np.float32)
        return joint_quat
