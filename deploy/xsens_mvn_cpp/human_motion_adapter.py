from __future__ import annotations

from typing import Sequence

import numpy as np

from deploy.utils.math_func import quat_inv, quat_mul
from deploy.xsens_mvn_cpp.types import HumanMotionSample, XSENS_TO_HUMAN_JOINT


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
        self.missing_position = np.asarray(missing_position, dtype=np.float32)
        self.missing_quaternion_wxyz = np.asarray(
            missing_quaternion_wxyz, dtype=np.float32
        )
        self.parent_indices = self._build_parent_indices()

    def to_human_sample(self, frame) -> HumanMotionSample:
        """Convert one latest raw frame into a `HumanMotionSample`.

        Preconditions:
        - `frame` is a snapshot returned by `xsens_mvn_cpp_py`.

        Postconditions:
        - Segment world positions and quaternions are copied to numpy arrays.
        - Joint angles are intentionally ignored in this first online human path.
        """

        segments_by_name = {segment.name: segment for segment in frame.segments}
        positions = np.zeros((len(self.desired_joint_names), 3), dtype=np.float32)
        quaternions = np.zeros((len(self.desired_joint_names), 4), dtype=np.float32)
        valid_mask = np.zeros(len(self.desired_joint_names), dtype=bool)

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

        joint_quaternions = self._derive_joint_quaternions(quaternions, valid_mask)

        frame_time = int(getattr(frame, "frame_time", 0))
        sequence = int(getattr(frame, "sequence", 0))
        return HumanMotionSample(
            joint_names=list(self.desired_joint_names),
            human_body_pos_w=positions,
            human_body_quat_w=quaternions,
            human_joint_quat=joint_quaternions,
            valid_mask=valid_mask,
            timestamp_ns=frame_time * 1_000_000,
            frame_id=f"xsens_raw:{sequence}",
        )

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

    def _derive_joint_quaternions(
        self,
        human_body_quat_w: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Derive local joint quaternions from parent and child world quaternions.

        Preconditions:
        - `human_body_quat_w` is ordered by `desired_joint_names`.
        - `valid_mask` marks entries with valid world segment poses.

        Postconditions:
        - Root entries use their world quaternion.
        - Non-root entries use `inverse(parent_world) * child_world`.
        - Missing entries use the configured identity/default quaternion.
        """

        joint_quat = np.tile(
            self.missing_quaternion_wxyz,
            (len(self.desired_joint_names), 1),
        ).astype(np.float32)

        for index, parent_index in enumerate(self.parent_indices):
            if not valid_mask[index]:
                continue
            if parent_index < 0:
                joint_quat[index] = human_body_quat_w[index]
                continue
            if not valid_mask[parent_index]:
                continue

            parent = human_body_quat_w[parent_index][None, :]
            child = human_body_quat_w[index][None, :]
            joint_quat[index] = quat_mul(quat_inv(parent), child)[0].astype(np.float32)
        return joint_quat

    def _build_parent_indices(self) -> np.ndarray:
        """Build desired-joint parent indices for local quaternion derivation.

        Preconditions:
        - `desired_joint_names` uses the deployment human joint naming convention.

        Postconditions:
        - Returns a shape `(num_joints,)` int32 array with `-1` for roots.
        """

        parent_by_name = {
            "Hips": -1,
            "Spine1": "Hips",
            "Spine2": "Spine1",
            "Chest": "Spine2",
            "Neck1": "Chest",
            "Neck2": "Neck1",
            "Head": "Neck2",
            "HeadEnd": "Head",
            "LeftShoulder": "Chest",
            "LeftArm": "LeftShoulder",
            "LeftForeArm": "LeftArm",
            "LeftHand": "LeftForeArm",
            "RightShoulder": "Chest",
            "RightArm": "RightShoulder",
            "RightForeArm": "RightArm",
            "RightHand": "RightForeArm",
            "LeftLeg": "Hips",
            "LeftShin": "LeftLeg",
            "LeftFoot": "LeftShin",
            "LeftToeBase": "LeftFoot",
            "LeftToeEnd": "LeftToeBase",
            "RightLeg": "Hips",
            "RightShin": "RightLeg",
            "RightFoot": "RightShin",
            "RightToeBase": "RightFoot",
            "RightToeEnd": "RightToeBase",
        }
        parent_indices = np.full(len(self.desired_joint_names), -1, dtype=np.int32)
        for index, joint_name in enumerate(self.desired_joint_names):
            parent_name = parent_by_name.get(joint_name, -1)
            if parent_name != -1 and parent_name in self.desired_joint_names:
                parent_indices[index] = self.desired_joint_names.index(parent_name)
        return parent_indices
