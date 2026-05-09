from deploy.deploy_mujoco.deploy_g1_mujoco import simulator
from deploy.utils.cfg import cfg
from deploy.utils.math_func import (
    matrix_from_quat,
    normalize,
    quat_apply_inverse,
    quat_mul,
    subtract_frame_transforms,
)
from deploy.xsens_mvn_cpp import (
    OnlineHumanMotionLoader,
    XsensPybindLatestFrameSource,
    XsensRawFrameHumanMotionAdapter,
)

import mujoco
import numpy as np
import time


class OnlineHumanMotionSimulator(simulator):
    ONLINE_HUMAN_ANCHOR_YAW_ALIGNMENT_DEGREES = 0.0
    XSENS_TO_SOMA_ANCHOR_BASIS_QUAT_W = np.asarray(
        [0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)],
        dtype=np.float32,
    )

    def __init__(
        self,
        xsens_udp_port=8001,
        xsens_receive_timeout_ms=2,
        xsens_max_datagram_size=5000,
        online_init_timeout_s=5.0,
    ):
        super().__init__()
        self.online_human_anchor_yaw_alignment_quat_w = self._yaw_quat_wxyz(
            self.ONLINE_HUMAN_ANCHOR_YAW_ALIGNMENT_DEGREES
        )
        self.online_human_loader = OnlineHumanMotionLoader(
            source=XsensPybindLatestFrameSource(
                udp_port=xsens_udp_port,
                receive_timeout_ms=xsens_receive_timeout_ms,
                max_datagram_size=xsens_max_datagram_size,
            ),
            adapter=XsensRawFrameHumanMotionAdapter(cfg.desire_human_joint_names),
            window_size=self.motion.window_size,
            initialization_timeout_s=online_init_timeout_s,
        )
        self.online_human_loader.start()
        self.online_human_loader.initialize()

    def close_online_receiver(self):
        self.online_human_loader.close()

    def _latest_online_human_sample(self):
        return self.online_human_loader.latest_sample()

    def _online_human_window(self, window_size):
        return self.online_human_loader.window()

    def policy_loop(self):
        policy_loop_start = time.perf_counter()
        self.online_human_loader.refresh()
        super().policy_loop()
        self._sleep_until_next_policy_tick(policy_loop_start)

    def sim_loop(self):
        for _ in range(self.control_decimation):
            if not cfg.motion_play:
                tau = self._PD_control(self.target_dof_pos)
                self.d.ctrl[:] = tau
            if not self.paused:
                self.prev_qpos = self.d.qpos.copy()
                self.set_camera()
                mujoco.mj_step(self.m, self.d)

    def _sleep_until_next_policy_tick(self, policy_loop_start):
        elapsed_s = time.perf_counter() - policy_loop_start
        sleep_s = self.policy_dt - elapsed_s
        if sleep_s > 0.0:
            time.sleep(sleep_s)

    def draw_current_human_skeleton(self):
        sample = self._latest_online_human_sample()
        if sample is None:
            return super().draw_current_human_skeleton()
        self.draw_human_skeleton(
            sample.human_body_pos_w,
            rotations=sample.human_body_quat_w,
            show_axes=self.show_human_skeleton_axes,
        )

    def _obs_ref_human_anchor_rot6d_in_sim_anchor(self):
        sample = self._latest_online_human_sample()
        if sample is None:
            return super()._obs_ref_human_anchor_rot6d_in_sim_anchor()

        self.pin.mujoco_to_pinocchio(
            self.d.qpos[7:],
            base_pos=self.d.qpos[0:3],
            base_quat=self.d.qpos[3:7][[1, 2, 3, 0]],
        )
        sim_robot_anchor_quat_w = np.expand_dims(
            self.pin.get_link_quaternion(cfg.motion_reference_body), axis=0
        )
        ref_human_anchor_quat_w = sample.human_body_quat_w[
            self.human_anchor_body_index, :
        ][None, :]
        ref_human_anchor_quat_w = self._align_online_human_anchor_quat(
            ref_human_anchor_quat_w
        )
        _, ref_human_anchor_quat_in_sim_anchor = subtract_frame_transforms(
            np.zeros((1, 3), dtype=np.float32),
            sim_robot_anchor_quat_w,
            np.zeros((1, 3), dtype=np.float32),
            ref_human_anchor_quat_w,
        )
        mat = matrix_from_quat(ref_human_anchor_quat_in_sim_anchor)
        return mat[..., :2].reshape(mat.shape[0], -1)

    def _obs_actor_ref_human_fsq_feature_window(self):
        window_size = self.motion.window_size
        window = self._online_human_window(window_size)
        if window is None:
            return super()._obs_actor_ref_human_fsq_feature_window()

        num_envs = 1
        num_human_bodies = len(self.fsq_human_body_indexes)
        human_anchor_quat = self._align_online_human_anchor_quat(
            window.human_body_quat_w[:, self.human_anchor_body_index][None, ...]
        )
        human_anchor_rot6d = self._rot6d_from_quat(human_anchor_quat)
        human_anchor_pos = window.human_body_pos_w[:, self.human_anchor_body_index][
            None, ...
        ]
        human_body_pos = window.human_body_pos_w[:, self.fsq_human_body_indexes, :][
            None, ...
        ]

        ref_human_body_pos_from_ref_anchor_w = human_body_pos - human_anchor_pos[
            :, :, None, :
        ]
        human_anchor_quat_w = np.broadcast_to(
            human_anchor_quat[:, :, None, :],
            (num_envs, window_size, num_human_bodies, 4),
        )
        ref_human_body_pos_in_ref_anchor = quat_apply_inverse(
            human_anchor_quat_w.reshape(-1, 4),
            ref_human_body_pos_from_ref_anchor_w.reshape(-1, 3),
        ).reshape(num_envs, window_size, -1)

        actor_human_feature = np.concatenate(
            (
                human_anchor_rot6d,
                ref_human_body_pos_in_ref_anchor,
            ),
            axis=-1,
        )
        return actor_human_feature.reshape(-1)

    def _align_online_human_anchor_quat(self, anchor_quat_w):
        basis_quat = np.broadcast_to(
            self.XSENS_TO_SOMA_ANCHOR_BASIS_QUAT_W,
            anchor_quat_w.shape,
        )
        yaw_quat = np.broadcast_to(
            self.online_human_anchor_yaw_alignment_quat_w,
            anchor_quat_w.shape,
        )
        aligned_quat = quat_mul(anchor_quat_w, basis_quat)
        aligned_quat = quat_mul(yaw_quat, aligned_quat)
        return normalize(aligned_quat).astype(np.float32)

    @staticmethod
    def _rot6d_from_quat(quaternions):
        quaternions = np.asarray(quaternions)
        mat = matrix_from_quat(quaternions)
        return mat[..., :2].reshape(mat.shape[:-2] + (6,))

    @staticmethod
    def _yaw_quat_wxyz(yaw_degrees):
        yaw_radians = np.deg2rad(yaw_degrees)
        half_yaw = yaw_radians * 0.5
        return np.asarray(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float32,
        )


if __name__ == "__main__":
    s = OnlineHumanMotionSimulator()
    try:
        s.run()
    finally:
        s.close_online_receiver()
