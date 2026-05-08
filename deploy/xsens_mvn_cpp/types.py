from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HumanMotionSample:
    """One sampled human-motion frame for online consumers.

    Preconditions:
    - Array fields are ordered by `joint_names`.

    Postconditions:
    - The sample is immutable at the dataclass level.
    """

    joint_names: list[str]
    human_body_pos_w: np.ndarray
    human_body_quat_w: np.ndarray
    human_joint_quat: np.ndarray
    human_joint_angles: np.ndarray
    valid_mask: np.ndarray
    joint_angle_valid_mask: np.ndarray
    timestamp_ns: int = 0
    frame_id: str = ""
    source_sample_counter: int = 0
    source_datagram_sequence: int = 0


@dataclass(frozen=True)
class HumanMotionWindow:
    """Fixed-size window of sampled human-motion frames.

    Preconditions:
    - First dimension of each array is the window dimension.

    Postconditions:
    - The window is immutable at the dataclass level.
    """

    joint_names: list[str]
    human_body_pos_w: np.ndarray
    human_body_quat_w: np.ndarray
    human_joint_quat: np.ndarray
    human_joint_angles: np.ndarray
    valid_mask: np.ndarray
    joint_angle_valid_mask: np.ndarray


XSENS_TO_HUMAN_JOINT = {
    "Hips": "pelvis",
    "Spine1": "l5",
    "Spine2": "l3",
    "Chest": "t8",
    "Neck1": "neck",
    "Neck2": "neck",
    "Head": "head",
    "HeadEnd": "head",
    "LeftShoulder": "left_shoulder",
    "LeftArm": "left_upper_arm",
    "LeftForeArm": "left_forearm",
    "LeftHand": "left_hand",
    "RightShoulder": "right_shoulder",
    "RightArm": "right_upper_arm",
    "RightForeArm": "right_forearm",
    "RightHand": "right_hand",
    "LeftLeg": "left_upper_leg",
    "LeftShin": "left_lower_leg",
    "LeftFoot": "left_foot",
    "LeftToeBase": "left_toe",
    "LeftToeEnd": "left_toe",
    "RightLeg": "right_upper_leg",
    "RightShin": "right_lower_leg",
    "RightFoot": "right_foot",
    "RightToeBase": "right_toe",
    "RightToeEnd": "right_toe",
}


HUMAN_JOINT_TO_XSENS_JOINT_SEGMENTS = {
    "Hips": (1, 1),
    "Spine1": (1, 2),
    "Spine2": (2, 3),
    "Chest": (4, 5),
    "Neck1": (5, 6),
    "Neck2": (5, 6),
    "Head": (6, 7),
    "HeadEnd": (6, 7),
    "LeftShoulder": (5, 12),
    "LeftArm": (12, 13),
    "LeftForeArm": (13, 14),
    "LeftHand": (14, 15),
    "RightShoulder": (5, 8),
    "RightArm": (8, 9),
    "RightForeArm": (9, 10),
    "RightHand": (10, 11),
    "LeftLeg": (1, 20),
    "LeftShin": (20, 21),
    "LeftFoot": (21, 22),
    "LeftToeBase": (22, 23),
    "LeftToeEnd": (22, 23),
    "RightLeg": (1, 16),
    "RightShin": (16, 17),
    "RightFoot": (17, 18),
    "RightToeBase": (18, 19),
    "RightToeEnd": (18, 19),
}
