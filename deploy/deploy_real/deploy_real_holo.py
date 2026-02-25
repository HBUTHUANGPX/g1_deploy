# from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config
import onnx
import onnxruntime
import json
LEGGED_GYM_ROOT_DIR = "/home/unitree/unitree_rl_gym-main"

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        # Initialize the policy network
        self.saved_low_state = {
            'joint_positions':[],
            'joint_velocities':[],
            'joint_efforts':[],
        }
        self.saved_low_cmd = []
        def load_onnx_policy(onnx_path: str):
            onnx_policy_session = onnxruntime.InferenceSession(onnx_path)
            input_names = [inp.name for inp in onnx_policy_session.get_inputs()]
            output_names = [out.name for out in onnx_policy_session.get_outputs()]

            onnx_input_names = input_names
            onnx_output_names = output_names

            # Extract metadata from ONNX model (hard fault if fails)
            onnx_model = onnx.load(onnx_path)
            metadata = {}
            for prop in onnx_model.metadata_props:
                metadata[prop.key] = json.loads(prop.value)

            onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
            onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

            print("onnx pd control kp kd:",onnx_kp,onnx_kd)
            def policy_act(actor_obs):
                obs_dict = {
                    'actor_obs': actor_obs.numpy().astype(np.float32)
                }
                input_feed = {name: obs_dict[name] for name in onnx_input_names}
                outputs = onnx_policy_session.run(onnx_output_names, input_feed)
                return outputs[0]  # just return outputs[0] as only "action" is needed
            return policy_act, onnx_kp, onnx_kd
        
        # self.policy = torch.jit.load(config.policy_path)
        self.policy, self.policy_kp, self.policy_kd = load_onnx_policy(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.tau = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.phase = torch.tensor([0.0, np.pi], dtype=torch.float64)
        self.policy_default_angles = np.array([
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            0.0, 0.0, 0.0,
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
            0.2,-0.2,  0.0, 0.6, 0.0, 0.0, 0.0,
        ])
        self.standing_mode = False
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
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
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
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
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
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

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        whole_body_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx 
        for i in range(len(whole_body_idx)):
            self.qj[i] = self.low_state.motor_state[whole_body_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[whole_body_idx[i]].dq
            self.tau[i] = self.low_state.motor_state[whole_body_idx[i]].ddq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        def build_sac_obs(qpos_, qvel_, quat, ang_vel, last_action, cmd, default_angles, phase,standing_mode=False):
            phase_dt = 2 * torch.pi * (1/50) * 1.0 # control dt : 0.02s , gait frequency : 1.0Hz
            if (np.linalg.norm(cmd[:2]) < 0.1 and np.abs(cmd[2]) < 0.1) :  # 站立模式
                phase = torch.tensor([np.pi, np.pi], dtype=torch.float64)
                standing_mode = True
            elif standing_mode:  # 从站立模式切换到行走模式
                phase = torch.tensor([0.0, np.pi], dtype=torch.float64)
                standing_mode = False
            else:
                phase_tp1 = phase + phase_dt
                phase = (torch.fmod(phase_tp1 + torch.pi, 2 * torch.pi) - torch.pi)
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)
            obs = np.zeros(100, dtype=np.float64)
            qpos = (qpos_ - default_angles)
            qvel = (qvel_ * 0.05)
            
            gravity_orientation = get_gravity_orientation(quat) 
            # ['actions', 'base_ang_vel', 'command_ang_vel', 'command_lin_vel', 'cos_phase', 'dof_pos', 'dof_vel', 'projected_gravity', 'sin_phase']
            obs[0:num_actions] = last_action[0:num_actions]
            obs[num_actions : num_actions + 3] =  ang_vel * 0.25
            obs[num_actions + 3 : num_actions + 6] = np.array([cmd[2], cmd[0], cmd[1]])
            obs[num_actions + 6 : num_actions + 8] = cos_phase.numpy()
            obs[num_actions + 8 : num_actions + 8 + num_actions] = qpos[0: num_actions]
            obs[num_actions + 8 + num_actions : num_actions + 8 + 2 * num_actions] = qvel[0: num_actions]
            obs[num_actions + 8 + 2 * num_actions : num_actions + 8 + 2 * num_actions + 3] = gravity_orientation
            obs[num_actions + 8 + 2 * num_actions + 3 : num_actions + 8 + 2 * num_actions + 5] = sin_phase.numpy()
            # obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            return obs,phase,standing_mode
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()

        self.cmd[0] = self.remote_controller.ly * 1.0
        self.cmd[1] = self.remote_controller.lx * -1 * 1.0
        self.cmd[2] = self.remote_controller.rx * -1 * 1.0
        # in sac cmd is in order: ang_vel, lin_vel_x, lin_vel_y
        # self.cmd = np.array([self.cmd[2], self.cmd[0], self.cmd[1]])
        # print("self.cmd", self.cmd)

        num_actions = self.config.num_actions
        self.obs, self.phase, self.standing_mode = build_sac_obs(
            qpos_ = qj_obs,
            qvel_ = dqj_obs,
            quat = quat,
            ang_vel = ang_vel,
            last_action = self.action,
            cmd = self.cmd,
            default_angles = self.policy_default_angles,
            phase = self.phase,
            standing_mode = self.standing_mode,
        )
        # print("self.obs:", self.obs)
        # self.obs *= 0.0
        # print("self.obs:", self.obs)

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor) #.detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.policy_default_angles + self.action * self.config.action_scale
        target_dof_pos = target_dof_pos.squeeze(0)
        # print("target_dof_pos:", target_dof_pos, target_dof_pos.shape)
        # print("self.action:", self.action, self.action.shape)
        # Build low cmd for whole body
        for i in range(len(whole_body_idx)):
            motor_idx = whole_body_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.policy_kp[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.policy_kd[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        # save
        self.send_cmd(self.low_cmd)
        # save
        self.saved_low_cmd.append(self.action.copy())
        self.saved_low_state['joint_positions'].append(self.qj.copy())
        self.saved_low_state['joint_velocities'].append(self.dqj.copy())
        self.saved_low_state['joint_efforts'].append(self.tau.copy())
        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()
    controller.saved_low_cmd = []
    controller.saved_low_state = {
            'joint_positions':[],
            'joint_velocities':[],
            'joint_efforts':[],
        }
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
    # save
    np.savez('action.npz', action = np.array(controller.saved_low_cmd))
    # joint_positions
    np.savez('joint_state.npz', joint_positions = np.array(controller.saved_low_state['joint_positions']),
    joint_velocities = np.array(controller.saved_low_state['joint_velocities']),
    joint_efforts = np.array(controller.saved_low_state['joint_efforts']))
    print("Exit")
