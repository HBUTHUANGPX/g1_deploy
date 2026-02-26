from deploy.utils.pinocchio_func import pin_mj
from deploy.utils.video_recorder import VideoRecorder
from deploy.utils.math_func import *
from deploy.utils.cfg import cfg, current_path
from deploy.utils.infer import infere

import numpy as np
import time
import os
from typing import Union

from config import Config
from common.remote_controller import RemoteController, KeyMap
from common.command_helper import (
    create_damping_cmd,
    create_zero_cmd,
    init_cmd_hg,
    init_cmd_go,
    MotorMode,
)
from common.rotation_helper import get_gravity_orientation, transform_imu_data

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.utils.crc import CRC

np.set_printoptions(precision=16, linewidth=100, threshold=np.inf, suppress=True)


class mini_g1_real:
    def __init__(self, config: Config):
        print("==mini_g1_real init==")
        self.config = config
        self.remote_controller = RemoteController()
        self.motor_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        self.qj = np.zeros(len(self.motor_idx), dtype=np.float32)
        self.dqj = np.zeros(len(self.motor_idx), dtype=np.float32)
        self.target_dof_pos = None
        self.counter = 0

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(
                config.lowstate_topic, LowStateHG
            )
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(
                config.lowstate_topic, LowStateGo
            )
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = (
            self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        )
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate(
            (self.config.default_angles, self.config.arm_waist_target), axis=0
        )
        dof_size = len(dof_idx)

        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = (
                    init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                )
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def perpare_data(self):
        # Get the current joint position and velocity
        
        for i in range(len(self.motor_idx)):
            self.qj[i] = self.low_state.motor_state[
                self.motor_idx[i]
            ].q
            self.dqj[i] = self.low_state.motor_state[
                self.motor_idx[i]
            ].dq

        # imu_state quaternion: w, x, y, z
        self.quat = np.array([1,0,0,0],dtype=np.float32)
        # self.quat = np.array(self.low_state.imu_state.quaternion,dtype=np.float32)
        self.ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[
                self.config.arm_waist_joint2motor_idx[0]
            ].q
            waist_yaw_omega = self.low_state.motor_state[
                self.config.arm_waist_joint2motor_idx[0]
            ].dq
            self.quat, self.ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=self.quat,
                imu_omega=self.ang_vel,
            )

    def update_cmd(self, target_pos=None, kps=None, kds=None):
        dof_idx = (
            self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        )
        dof_size = len(dof_idx)
        if target_pos is None:
            _target_pos = [0] * (dof_size)
        else:
            _target_pos = target_pos
        if kps is None:
            _kps = self.config.kps + self.config.arm_waist_kps
        else:
            _kps = kps

        if kds is None:
            _kds = self.config.kds + self.config.arm_waist_kds
        else:
            _kds = kds

        for j in range(dof_size):
            motor_idx = dof_idx[j]
            self.low_cmd.motor_cmd[motor_idx].q = _target_pos[j]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = _kps[j]
            self.low_cmd.motor_cmd[motor_idx].kd = _kds[j]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)

    def run(self):
        self.counter += 1
        self.perpare_data()
        # =========================
        #         infer
        # =========================
        # send the command
        self.update_cmd(self.target_dof_pos)
        time.sleep(self.config.control_dt)

class real(infere, mini_g1_real):
    def __init__(self, config: Config):
        super().__init__()
        super(infere,self).__init__(config)

    def run(self):
        self.counter += 1
        self.perpare_data()
        # =========================
        #         infer
        # =========================
        self.minimum_infer()
        
        if self.time_step >= self.motion.time_step_total:
            self.time_step = 1
        # send the command
        self.update_cmd(self.target_dof_pos,kps = self.P_gains,kds = self.D_gains)
        time.sleep(self.config.control_dt)

    def _obs_motion_joint_pos_command(self):
        return np.copy(self.motion.joint_pos[int(self.time_step)])

    def _obs_motion_joint_vel_command(self):
        return np.copy(self.motion.joint_vel[int(self.time_step)])

    def _obs_motion_ref_ori_b(self):
        self.pin.mujoco_to_pinocchio(
            self.qj,
            base_pos=np.array([0,0,0],dtype=np.float32),
            base_quat=self.quat[[1, 2, 3, 0]],
        )
        _quat = self.pin.get_link_quaternion(cfg.motion_reference_body)
        self.robot_ref_quat_w = np.expand_dims(_quat, axis=0)  # shape [n,4]
        self.ref_quat_w = self.motion.body_quat_w[
            int(self.time_step), cfg.motion_body_names.index(cfg.motion_reference_body), :
        ]  # shape [n,4]
        q01 = self.robot_ref_quat_w
        q02 = self.ref_quat_w
        if q02 is not None and q02.ndim == 1:
            q02 = np.expand_dims(q02, axis=0)
        q10 = quat_inv(q01)
        if q02 is not None:
            q12 = quat_mul(q10, q02)
        else:
            q12 = q10
        mat = matrix_from_quat(q12)
        motion_ref_ori_b = mat[..., :2].reshape(mat.shape[0], -1)  # shape [n,6]
        return motion_ref_ori_b

    def _obs_base_ang_vel(self):
        return self.ang_vel * 0

    def _obs_joint_pos(self):
        return (self.qj - self.default_pos)[self.mujoco2isaac_sim_index]

    def _obs_joint_vel(self):
        return self.dqj[self.mujoco2isaac_sim_index] * 0

    def _obs_actions(self):
        return self.action

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net", 
        type=str, default="eth0", help="network interface")
    parser.add_argument(
        "--config",
        type=str,
        help="config file name in the configs folder",
        default="g1.yaml",
    )
    args = parser.parse_args()

    # Load config
    config_path = current_path + f"/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = real(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
