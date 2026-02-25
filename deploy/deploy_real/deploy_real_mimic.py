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
from common.rotation_helper import get_gravity_orientation, transform_imu_data, qinverse, qmult, quat2mat
from common.remote_controller import RemoteController, KeyMap
from config import Config
import onnx
import onnxruntime
import json
# from transforms3d.quaternions import quat2mat, qinverse, qmult

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
        def load_motion_npz(mimic_motion_path: str):
            # Load motion data from a file (assuming .npz format for this example)
            if mimic_motion_path is None:
                mimic_motion_path = os.path.join(os.path.dirname(__file__), "tmp",  "motion.npz")
            motion_data = np.load(mimic_motion_path)
            self.cmd_joint_pos_buffer = motion_data['joint_pos']
            self.cmd_joint_vel_buffer = motion_data['joint_vel']
            self.cmd_body_pos_w_buffer = motion_data['body_pos_w']
            self.cmd_body_quat_w_buffer = motion_data['body_quat_w']
            self.cmd_body_lin_vel_w_buffer = motion_data['body_lin_vel_w']
            self.cmd_body_ang_vel_w_buffer = motion_data['body_ang_vel_w']
            self.num_frames = self.cmd_joint_pos_buffer.shape[0]
            # print("motion fps:", motion_data['fps'] )
            self.fps = motion_data['fps'].item() if 'fps' in motion_data else 30.0

        def load_onnx_policy(policy_path: str):
            # load onnx, extract kp, kd, action scale etc
            assert policy_path.endswith('.onnx'), "Only .onnx policy files are supported in this example."
            # Placeholder for actual policy loading logic
            onnx_policy_session = onnxruntime.InferenceSession(policy_path)
            input_names = [inp.name for inp in onnx_policy_session.get_inputs()]
            output_names = [out.name for out in onnx_policy_session.get_outputs()]
            onnx_model = onnx.load(policy_path)
            metadata = {}
            for prop in onnx_model.metadata_props:
                metadata[prop.key] = prop.value
            kp = metadata['joint_stiffness'].split(',')
            kd = metadata['joint_damping'].split(',') 
            self.onnx_kp = np.array([float(k) for k in kp], dtype=np.float32) 
            self.onnx_kd = np.array([float(k) for k in kd], dtype=np.float32)
            self.joint_names = metadata['joint_names'].split(',') if "joint_names" in metadata else None
            print("Loaded joint names from onnx metadata:", self.joint_names)
            self.default_joint_pos = np.array(metadata['default_joint_pos'].split(','), dtype=np.float32) if "default_joint_pos" in metadata else None
            self.action_scale = np.array(metadata['action_scale'].split(','), dtype=np.float32) if "action_scale" in metadata else self.action_scale
            # map isaaclab joint
            print("onnx pd control kp kd:",self.onnx_kp,self.onnx_kd)
            print("action scale:", self.action_scale)
            def policy_act(obs_dict):
                input_feed = {name: obs_dict[name] for name in input_names }
                action = onnx_policy_session.run(output_names, input_feed)[0]
                return action.squeeze(0)
            # remap # Attention! you have to fill it before running the script
            self.isaaclab2mujoco_idx = [0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28]
            self.mujoco2isaaclab_idx = [0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28]
            # remap onnx kp and onnx kd, default joint pos, action scale
            self.onnx_kp = self.onnx_kp[self.isaaclab2mujoco_idx]
            self.onnx_kd = self.onnx_kd[self.isaaclab2mujoco_idx]
            self.default_joint_pos = self.default_joint_pos[self.isaaclab2mujoco_idx]
            self.action_scale = self.action_scale[self.isaaclab2mujoco_idx]
            # pitch id = 0, 6
            self.onnx_kp[[0,2,6,8]] *= 2.0
            self.onnx_kd[[0,2,6,8]] *= 1.5
            # ankle pitch/roll id: 4,5, 10,11
            self.onnx_kp[[4,5,10,11]] *= 2.0
            self.onnx_kd[[4,5,10,11]] *= 1.2

            # self.onnx_kd
            for i in range(len(self.joint_names)):
                joint_name = self.joint_names[self.isaaclab2mujoco_idx[i]]
                print("Joint:", joint_name, "Kp:", self.onnx_kp[i], "Kd", self.onnx_kd[i], "default_pos:", self.default_joint_pos[i])
            self.policy = policy_act
            
        load_onnx_policy(config.policy_path)
        load_motion_npz(config.motion_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.tau = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.default_leg_pose = self.default_joint_pos[:12]
        self.default_arm_pose = self.default_joint_pos[12:]
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.counter = 0

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
        default_pos = self.default_joint_pos.copy()
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
                self.low_cmd.motor_cmd[motor_idx].q = self.default_leg_pose[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.default_arm_pose[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        self.policy_init_base_quat = self.low_state.imu_state.quaternion
        self.policy_init_base_quat_inv = qinverse(self.policy_init_base_quat)

            

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
        def build_mimic_obs(qpos_, qvel_, quat, ang_vel, last_action):
            # quat in w,x,y,z format
            obs = np.zeros(config.num_obs, dtype=np.float64)
            qpos = (qpos_ - self.default_joint_pos)
            qvel = (qvel_ * 1.0)
            
            # remap qpos, qvel
            qpos = qpos[self.mujoco2isaaclab_idx]
            qvel = qvel[self.mujoco2isaaclab_idx]
            # [ cmd_joint_pos, cmd_joint_vel, cmd_base_ori_mat, base_ang_vel, joint_pos, joint_vel, actions]
            cmd_joint_pos = self.cmd_joint_pos_buffer[self.counter].copy()
            cmd_joint_vel = self.cmd_joint_vel_buffer[self.counter].copy()
            cmd_base_ori = self.cmd_body_quat_w_buffer[self.counter][0].copy()
            # quat 2 mat with 2 cols
            # we need to record the initial robot quat, then apply to motion zip or robot quat
            robot_quat = qmult(self.policy_init_base_quat_inv, quat)
            # print("robot_quat", robot_quat)
            robot_quat_inv = qinverse(robot_quat)
            quat_diff = qmult(robot_quat_inv, cmd_base_ori)
            cmd_base_ori_mat = quat2mat(quat_diff)[:,:2].reshape(-1)
            obs = np.concatenate(
                [
                    cmd_joint_pos,
                    cmd_joint_vel,
                    cmd_base_ori_mat,
                    ang_vel.squeeze(0),
                    qpos,
                    qvel,
                    last_action,
                ], dtype=np.float32
            )
            return obs
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()

        num_actions = self.config.num_actions
        self.obs = build_mimic_obs(
            qpos_ = qj_obs,
            qvel_ = dqj_obs,
            quat = quat,
            ang_vel = ang_vel,
            last_action = self.action,
        )
        

        # Get the action from the policy network
        obs_dict = {
            'obs': self.obs.reshape(1,-1),
            'time_step': np.array([[self.counter]], dtype=np.float32)
        }
        self.action = self.policy(obs_dict)
        #  remap action
        action_ = self.action[self.isaaclab2mujoco_idx]
        target_dof_pos = self.default_joint_pos + action_ * self.action_scale
        # Build low cmd for whole body
        for i in range(len(whole_body_idx)):
            motor_idx = whole_body_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.onnx_kp[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.onnx_kd[i]
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
            if controller.counter > controller.num_frames:
                break
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
