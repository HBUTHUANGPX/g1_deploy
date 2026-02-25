import mujoco
import mujoco.viewer
import numpy as np
import os
import torch
import argparse
import onnxruntime
import onnx
from transforms3d.quaternions import quat2mat, qinverse, qmult
import time
class MimicSim2Sim:
    def __init__(self, motion_file_path, policy_path=None, xml_path=None):
        if not os.path.exists(motion_file_path):
            raise FileNotFoundError(f"Motion file not found: {motion_file_path}")
        self._load_motion_file(motion_file_path)
        self.current_frame = 0
        self._load_policy(policy_path)
        self._setup_simulation(xml_path)
        self._match_joints()

        # config
        self.num_dofs =  self.mj_data.qpos.shape[0] - 7  # assuming first 7 are base pos and orientation
        self.control_decimation = 10
        self.actions = np.zeros(self.num_dofs)


    def _match_joints(self):
        if self.joint_names is None:
            raise ValueError("Joint names not found in policy metadata.")
        mujoco_joint_set = set(self.mj_joint_names[1:])  # exclude base joints
        isaaclab_joint_set = set(self.joint_names)
        common_joints = mujoco_joint_set.intersection(isaaclab_joint_set)
        if len(common_joints) == 0:
            raise ValueError("No common joints found between MuJoCo model and policy.")
        # create mapping from isaaclab joint index to mujoco joint index
        self.mujoco2isaaclab_idx = []
        for joint_name in self.joint_names:
            if joint_name in common_joints:
                mujoco_idx = self.mj_joint_names.index(joint_name) - 1  # adjust for base joints
                self.mujoco2isaaclab_idx.append(mujoco_idx)
        self.mujoco2isaaclab_idx = np.array(self.mujoco2isaaclab_idx)
        self.isaaclab2mujoco_idx = np.argsort(self.mujoco2isaaclab_idx)

        if self.default_joint_pos is not None:
            self.default_joint_pos = self.default_joint_pos[self.isaaclab2mujoco_idx]

        if self.kp is not None:
            self.kp = self.kp[self.isaaclab2mujoco_idx]
            # print("Remapped kp:", self.kp)
        if self.kd is not None:
            self.kd = self.kd[self.isaaclab2mujoco_idx]

        if self.action_scale is not None and len(self.action_scale) > 1:
            self.action_scale = self.action_scale[self.isaaclab2mujoco_idx]


    def _load_policy(self, policy_path):
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
        self.kp = np.array([float(k) for k in kp], dtype=np.float32) 
        self.kd = np.array([float(k) for k in kd], dtype=np.float32)
        self.joint_names = metadata['joint_names'].split(',') if "joint_names" in metadata else None
        print("Loaded joint names from onnx metadata:", self.joint_names)
        self.default_joint_pos = np.array(metadata['default_joint_pos'].split(','), dtype=np.float32) if "default_joint_pos" in metadata else None
        self.action_scale = np.array(metadata['action_scale'].split(','), dtype=np.float32) if "action_scale" in metadata else self.action_scale
        # map isaaclab joint
        print("onnx pd control kp kd:",self.kp,self.kd)
        def policy_act(obs_dict):
            input_feed = {name: obs_dict[name] for name in input_names }
            action = onnx_policy_session.run(output_names, input_feed)[0]
            return action.squeeze(0)
        self.policy = policy_act
        return None

    # motion file path
    def _load_motion_file(self, motion_file_path):
        # Load motion data from a file (assuming .npz format for this example)
        if motion_file_path is None:
            motion_file_path = os.path.join(os.path.dirname(__file__), "tmp",  "motion.npz")
        motion_data = np.load(motion_file_path)
        self.cmd_joint_pos_buffer = motion_data['joint_pos']
        self.cmd_joint_vel_buffer = motion_data['joint_vel']
        self.cmd_body_pos_w_buffer = motion_data['body_pos_w']
        self.cmd_body_quat_w_buffer = motion_data['body_quat_w']
        self.cmd_body_lin_vel_w_buffer = motion_data['body_lin_vel_w']
        self.cmd_body_ang_vel_w_buffer = motion_data['body_ang_vel_w']
        self.num_frames = self.cmd_joint_pos_buffer.shape[0]
        print("motion fps:", motion_data['fps'] )
        self.fps = motion_data['fps'].item() if 'fps' in motion_data else 30.0


    def _setup_simulation(self, mj_xml_path):
        # Load the MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(mj_xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_xml = mj_xml_path
        self.mj_model.opt.timestep = (1/500)
        self.mj_joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        print("MuJoCo joint names:", self.mj_joint_names)

    def reset_simulation(self):

        self.mj_data.qpos[7:] = self.default_joint_pos.copy()
        self.mj_data.qvel[:] = 0.0
        self.mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # set base orientation (quaternion)

        if self.mj_data is not None:
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self.current_frame = 0

        else:
            raise RuntimeError("Simulation data is not initialized.")

    def step_simulation(self):
        self.obs = self.build_obs(self.actions)
        obs_dict = {"obs": self.obs.unsqueeze(0).numpy(), 'time_step': np.array([[self.current_frame]], dtype=np.float32)}
        self.actions = self.policy(obs_dict)
        if self.mj_data is not None:
            for _ in range(self.control_decimation):
                step_start_time = time.time()
                tau = self.pd_control(self.actions)
                self.mj_data.ctrl[:] = tau
                mujoco.mj_step(self.mj_model, self.mj_data)
                time_till_next_step = self.mj_model.opt.timestep - (time.time() - step_start_time)
                if time_till_next_step > 0:
                    time.sleep(time_till_next_step)

            self.current_frame += 1
            print(f"Current frame: {self.current_frame}/{self.num_frames}", end='\r')
            if self.current_frame >= self.num_frames:
                self.current_frame = 0
                self.reset_simulation()
        else:
            raise RuntimeError("Simulation data is not initialized.")

    def pd_control(self, actions):
        actions_ = actions[self.isaaclab2mujoco_idx].copy()
        joint_pos = self.mj_data.qpos[7:].copy()
        joint_vel = self.mj_data.qvel[6:].copy()
        target_qpos = actions_ * self.action_scale + self.default_joint_pos
        tau = self.kp * (target_qpos - joint_pos) - self.kd *  joint_vel
        return tau

    def build_obs(self, actions):
        if self.mj_data is not None:
            joint_pos = self.mj_data.qpos[7:].copy() - self.default_joint_pos
            joint_vel = self.mj_data.qvel[6:].copy()
            # remap
            joint_pos = joint_pos[self.mujoco2isaaclab_idx]
            joint_vel = joint_vel[self.mujoco2isaaclab_idx]

            # base_lin_vel = self.mj_data.qvel[0:3].copy()
            base_ang_vel = self.mj_data.qvel[3:6].copy()
            actions = actions.copy()
            cmd_joint_pos = self.cmd_joint_pos_buffer[self.current_frame].copy() 
            cmd_joint_vel = self.cmd_joint_vel_buffer[self.current_frame].copy()
            cmd_base_pos = self.cmd_body_pos_w_buffer[self.current_frame][0].copy()
            cmd_base_ori = self.cmd_body_quat_w_buffer[self.current_frame][0].copy()
            # quat to mat with 2 cols
            base_quat= self.mj_data.qpos[3:7].copy() # w,x,y,z
            base_quat_inv = qinverse(base_quat)
            quat_diff = qmult(base_quat_inv, cmd_base_ori)
            cmd_base_ori_mat = quat2mat(quat_diff)[:, :2].reshape(-1)
            base_pos_diff = cmd_base_pos - self.mj_data.qpos[0:3].copy()
            rotated_base_pos_diff = quat2mat(base_quat_inv).dot(base_pos_diff)
            obs = np.concatenate(
                [
                    cmd_joint_pos,
                    cmd_joint_vel,
                    # rotated_base_pos_diff,
                    cmd_base_ori_mat,
                    # base_lin_vel,
                    base_ang_vel,
                    joint_pos,
                    joint_vel,
                    actions,
                ]
            )
            return torch.tensor(obs, dtype=torch.float32)
        else:
            raise RuntimeError("Simulation data is not initialized.")

    def run(self):
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            self.reset_simulation()
            while viewer.is_running():
                self.step_simulation()
                viewer.sync()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, default=f"{os.path.dirname(__file__)}/tmp/motion.npz")
    parser.add_argument("--policy_file", type=str, default=f"{os.path.dirname(__file__)}/onnx/policy_wo_ref_base_pos.onnx")
    parser.add_argument("--mj_xml", type=str, default=f"{os.path.dirname(__file__)}/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml")
    args = parser.parse_args()

    sim = MimicSim2Sim(args.motion_file, args.policy_file, args.mj_xml)
    sim.run()