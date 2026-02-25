


from deploy.utils.observation_manager import SimpleObservationManager, TermCfg, GroupCfg
from deploy.utils.pinocchio_func import pin_mj
from deploy.utils.urdf_graph import UrdfGraph
from deploy.utils.motor_conf import *
from deploy.utils.motion_loader import MotionLoader
from deploy.utils.video_recorder import VideoRecorder
from deploy.utils.math_func import *
import onnxruntime as ort
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import copy

import mujoco.viewer
import mujoco



np.set_printoptions(precision=16, linewidth=100, threshold=np.inf, suppress=True)

class ObsCfg:
    """观测总配置：每个属性是一个 GroupCfg 实例。"""

    class PolicyCfg(GroupCfg):
        motion_joint_pos_command = TermCfg()
        motion_joint_vel_command = TermCfg()
        motion_ref_ori_b = TermCfg()
        # base_ang_vel = TermCfg(history_length=24)
        # joint_pos = TermCfg(history_length=24)
        # joint_vel = TermCfg(history_length=24)
        base_ang_vel = TermCfg()
        joint_pos = TermCfg()
        joint_vel = TermCfg()
        actions = TermCfg()

    policy = PolicyCfg()

current_path = os.getcwd()
print(current_path)
class cfg:
    simulator_dt = 0.002
    policy_dt = 0.02

    policy_path = (
        current_path
        + "/"
        + "deploy/deploy_policy/g1/2026-02-25_15-08-29_G1_slowly_walk"
        + "/policy.onnx"
    )
    asset_path = current_path + "/deploy/assets/unitree_g1"
    mjcf_path = asset_path + "/g1_29dof_rev_1_0.xml"
    urdf_path = asset_path + "/g1_29dof_mode_15.urdf"
    motion_file = (
        current_path
        + "/deploy/artifacts/g1/"
        + "xsens_bvh/251203/01_slowly_forward_walk_120Hz.npz"
    )
    only_leg_flag = False  # True, False
    with_wrist_flag = True  # True, False

    ###################################################
    # stiffness damping and joint maximum torqueparam #
    ###################################################
    leg_P_gains = [STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020] * 2
    leg_D_gains = [DAMPING_7520_14, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020] * 2
    leg_tq_max = [88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * (2)

    pelvis_P_gains = [STIFFNESS_7520_14, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020]
    pelvis_D_gains = [DAMPING_7520_14, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020]
    pelvis_tq_max = [88, 50, 50]

    arm_P_gains = [STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010] * (2)
    arm_D_gains = [DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_4010, DAMPING_4010] * (2)
    arm_tq_max = [25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0] * (2)

    ########################
    # joint maximum torque #
    ########################

    #####################
    # joint default pos #
    #####################
    leg_default_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * (2)
    pelvis_default_pos = [0.0] * (3)
    arm_default_pos = [0.0] * (7*2)

    ################
    # action param #
    ################
    action_clip = 10.0
    action_scale = 0.25

    action_num = 29
    print("action_num: ", action_num)
    #############
    # obs param #
    #############
    frame_stack = 1
    num_single_obs = 154 #1557 154

    ####################
    # motion play mode #
    ####################
    """
     if motion_play is true, robots in mujoco will set 
     qpos and qvel through the retargeting dataset 
    """
    motion_play = False  # False, True
    """
    if motion_play is true and sim_motion_play is true,
    robots in mujoco will set qpos and qvel through the 
    dataset recorded in isaac sim
    """
    sim_motion_play = False  # False, True,

    ###########################################
    # Data conversion of isaac sim and mujoco #
    ###########################################
    urdf_graph = UrdfGraph(urdf_path)
    isaac_sim_joint_name = urdf_graph.bfs_joint_order()

    isaac_sim_link_name = urdf_graph.bfs_link_order() # env.unwrapped.scene["robot"].body_names

    motion_body_names = [
    "pelvis",

    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",

    "torso_link",

    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

    motion_reference_body = "torso_link"


class simulator:
    policy: ort.InferenceSession

    def __init__(self):
        # Load robot model
        self.spec = mujoco.MjSpec.from_file(cfg.mjcf_path)
        # self._rehandle_xml()
        # self.m = self.spec.compile()
        self.pin = pin_mj(cfg.urdf_path)
        self.m = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
        self.d = mujoco.MjData(self.m)
        self._scene = mujoco.MjvScene(self.m, 100000)
        print(f"Number of actuators: {self.m.nu}")

        self.m.opt.timestep = cfg.simulator_dt
        self.paused = False
        self._init_robot_conf()
        self._init_policy_conf()
        self.change_id = 0
        self.video_recorder = VideoRecorder(
            path=current_path + "/deploy_mujoco/recordings",
            tag=None,
            video_name="video_0",
            fps=int(1 / cfg.policy_dt),
            compress=False,
        )
        self.data_save = []
        self.obs_manager = SimpleObservationManager(ObsCfg(), self)
        to = self.obs_manager.compute_group("policy", update_history=True)

    def motion_play(self):
        self.d.qpos[0:3] = (
            self.motion.body_pos_w[self.time_step, 7, :].detach().cpu().numpy()
        )
        q = self.motion.body_quat_w[self.time_step, 0, :].detach().cpu().numpy()[0, :]
        # self.d.qpos[3:7] = np.array([0, 0, 0, 1], dtype=np.float32)
        self.d.qpos[3:7] = q
        self.d.qpos[7 : 7 + len(self.default_pos)] = (
            self.motion.joint_pos[self.time_step].detach().cpu().numpy()
        )[:, self.isaac_sim2mujoco_index]
        self.d.qvel[0:3] = 0 * (
            self.motion.body_lin_vel_w[self.time_step, 0, :].detach().cpu().numpy()
        )
        self.d.qvel[3:6] = 0 * (
            self.motion.body_ang_vel_w[self.time_step, 0, :].detach().cpu().numpy()
        )
        self.d.qvel[6 : 6 + len(self.default_pos)] = (
            self.motion.joint_vel[self.time_step]
            .detach()
            .cpu()
            .numpy()[:, self.isaac_sim2mujoco_index]
        )
        mujoco.mj_forward(self.m, self.d)
        return

    def run(self):
        save_data_flag = 1
        self.counter = 0
        self.d.qpos[7 : 7 + len(self.default_pos)] = self.default_pos
        self.d.qpos[2] = 0.992
        mujoco.mj_forward(self.m, self.d)
        self.target_dof_pos = self.default_pos.copy()[: self.action_num]
        self.phase = 0
        # self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        if save_data_flag:
            i = 0
            if os.path.exists("data.csv"):
                os.remove("data.csv")
        self.viewer = mujoco.viewer.launch_passive(
            self.m, self.d, key_callback=self.key_callback
        )
        self.renderer = mujoco.renderer.Renderer(self.m, height=480, width=640)
        self.init_vel_geom(
            "Goal Vel: x: {:.2f}, y: {:.2f}, yaw: {:.2f},force_z:{:.2f}".format(
                self.cmd[0], self.cmd[1], self.cmd[2], 0.0
            )
        )
        self.prev_qpos = self.d.qpos

        first_flag = False

        log = {
            "fps": [50],
            "dof_names": [joint.name for joint in self.spec.joints][1:],
            "body_names": self.mujoco_all_body_names,
            "dof_positions": [],
            "dof_velocities": [],
            "dof_torque": [],
            "body_positions": [],
            "body_rotations": [],
            "body_linear_velocities": [],
            "body_angular_velocities": [],
            "qpos": [],
            "qvel": [],
            "xpos": [],
            "xquat": [],
            "cvel": [],
            "P_gain": [self.P_gains],
            "D_gain": [self.D_gains],
            "target_pos": [],
            "qfrc_actuator":[],
        }

        while self.viewer.is_running():
            if not first_flag:
                first_flag = True
                if cfg.motion_play:
                    self.motion_play()
                    self.time_step *= 0
                    if cfg.sim_motion_play:
                        self.time_step[:] = self.motion_play_isaac_sim() * 1.0
                else:
                    # self.motion_play()
                    ...
                mujoco.mj_step(self.m, self.d)
                self.viewer.sync()
            self.policy_loop()
            # print(self.time_step, self.motion.time_step_total)
            log["dof_positions"].append(np.copy(self.d.qpos[7:]))
            log["dof_velocities"].append(np.copy(self.d.qvel[6:]))
            log["dof_torque"].append(np.copy(self.d.qfrc_actuator[6:]))
            log["body_positions"].append(
                np.copy(self.d.xpos[self.mujoco_body_names_indices, :])
            )
            log["body_rotations"].append(
                np.copy(self.d.xquat[self.mujoco_body_names_indices, :])
            )
            log["body_linear_velocities"].append(
                np.copy(self.d.cvel[self.mujoco_body_names_indices, 0:3])
            )
            log["body_angular_velocities"].append(
                np.copy(self.d.cvel[self.mujoco_body_names_indices, 3:6])
            )
            log["qpos"].append(np.copy(self.d.qpos))
            log["qvel"].append(np.copy(self.d.qvel))
            log["xpos"].append(np.copy(self.d.xpos[self.mujoco_body_names_indices, :]))
            log["xquat"].append(
                np.copy(self.d.xquat[self.mujoco_body_names_indices, :])
            )
            log["cvel"].append(np.copy(self.d.cvel[self.mujoco_body_names_indices, :]))
            log["target_pos"].append(np.copy(self.target_dof_pos))
            log["qfrc_actuator"].append(np.copy(self.d.qfrc_actuator))
            # if self.time_step >= 50*60:
            if self.time_step >= self.motion.time_step_total:
                break
        for k in (
            "dof_positions",
            "dof_velocities",
            "body_positions",
            "body_rotations",
            "body_linear_velocities",
            "body_angular_velocities",
            "qpos",
            "qvel",
            "xpos",
            "xquat",
            "cvel",
            "qfrc_actuator"
        ):
            log[k] = np.stack(log[k], axis=0)
        np.savez(
            "/home/hpx/HPX_LOCO_2/mimic_baseline/deploy_mujoco/motion.npz", **log
        )
        # with open("data_save.pkl", "wb") as f:
        #     pickle.dump(self.data_save, f)
        print("stop")
        self.video_recorder.stop()

    def policy_loop(self):
        # print("="*(20))
        self.counter += 1
        # print(self.d.qvel[0])
        quat = self.d.qpos[3:7]
        omega = self.d.qvel[3:6]
        self.qpos = self.d.qpos[7:]
        self.P_n = self.qpos - self.default_pos
        self.V_n = self.d.qvel[6:]

        # if self.time_step >= 100:
        if self.time_step >= self.motion.time_step_total:
            self.time_step = 10

        if cfg.motion_play:
            if cfg.sim_motion_play:
                self.time_step[:] = self.motion_play_isaac_sim() * 1.0
            else:
                self.motion_play()
        else:
            # self.update_obs(self.time_step*0)
            self.update_obs(self.time_step)
            self.h2_action = self.h_action.copy()
            self.h_action = self.action.copy()
            self._policy_reasoning()
            # print(self.motion.joint_pos[self.time_step],"\r\n",self.r_joint_pos)
        action = (
            np.clip(
                copy.deepcopy(self.action[self.isaac_sim2mujoco_index]),
                -self.action_clip,
                self.action_clip,
            )
            * self.action_scale
            * self.tq_max
            / self.P_gains
            + self.default_pos
        )
        # print(self.action_scale
        #     * self.tq_max
        #     / self.P_gains)
        # {'.*_hip_roll_joint': 0.125, '.*_hip_yaw_joint': 0.08333333333333333, '.*_hip_pitch_joint': 0.25, '.*_knee_joint': 0.25, '.*_ankle_pitch_joint': 0.12857142857142856, '.*_ankle_roll_joint': 0.12857142857142856, 'pelvis_joint': 0.13291139240506328, 'head_yaw_joint': 0.21, 'head_pitch_joint': 0.21, '.*_shoulder_pitch_joint': 0.15, '.*_shoulder_roll_joint': 0.15, '.*_shoulder_yaw_joint': 0.08214285714285714, '.*_elbow_joint': 0.08214285714285714, '.*_forearm_yaw_joint': 0.10375000000000001, '.*_wrist_roll_joint': 0.041249999999999995, '.*_wrist_yaw_joint': 0.041249999999999995}
        target_q = action.clip(-self.action_clip, self.action_clip)
        # print(target_q)
        self.target_dof_pos = target_q  # + self.default_pos[: self.action_num]
        self.time_step += 1
        print(f"time_step: {self.time_step}")
        # self.time_step *= 0
        self.contact_force()
        self.sim_loop()
        # mujoco.mjr_render(self._viewport, self._scene, self._context)
        # im = self.read_pixels()
        # self.video_recorder(im)
        # 更新 Renderer 场景，使用查看器的相机和选项，使图像与窗口一致
        self.renderer.update_scene(
            self.d,
            camera=self.viewer.cam,  # 使用查看器的相机视图
            scene_option=self.viewer.opt,  # 使用查看器的渲染选项
        )

        # 捕获图像：返回 (height, width, 3) 的 uint8 NumPy 数组 (RGB)
        img = self.renderer.render()
        self.video_recorder(img)

        self.viewer.sync()
        self.update_vel_geom()

    def _obs_motion_joint_pos_command(self):
        return np.copy(self.motion.joint_pos[self.time_step])

    def _obs_motion_joint_vel_command(self):
        return np.copy(self.motion.joint_vel[self.time_step])

    def _obs_motion_ref_ori_b(self):
        self.pin.mujoco_to_pinocchio(
            self.d.qpos[7:],
            base_pos=self.d.qpos[0:3],
            base_quat=self.d.qpos[3:7][[1, 2, 3, 0]],
        )
        _quat = self.pin.get_link_quaternion(cfg.motion_reference_body)
        self.robot_ref_quat_w = torch.from_numpy(_quat).unsqueeze(0)  # shape [n,4]
        self.ref_quat_w = self.motion.body_quat_w[
            self.time_step, cfg.motion_body_names.index(cfg.motion_reference_body), :
        ]  # shape [n,4]
        q01 = self.robot_ref_quat_w
        q02 = self.ref_quat_w
        q10 = quat_inv(q01)
        if q02 is not None:
            q12 = quat_mul(q10, q02)
        else:
            q12 = q10
        mat = matrix_from_quat(q12)
        motion_ref_ori_b = mat[..., :2].reshape(mat.shape[0], -1)  # shape [n,6]
        return motion_ref_ori_b

    def _obs_base_ang_vel(self):
        return self.d.qvel[3:6]

    def _obs_joint_pos(self):
        return (self.d.qpos[7:] - self.default_pos)[self.mujoco2isaac_sim_index]
    
    def _obs_joint_vel(self):
        return self.d.qvel[6:][self.mujoco2isaac_sim_index]
    
    def _obs_actions(self):
        return self.action
    
    def update_obs(self, time_step):
        """
        +----------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (154,)) |
        +------------+--------------------------------+------------+
        |   Index    | Name                           |   Shape    |
        +------------+--------------------------------+------------+
        |     0      | command                        |   (58,)    |
        |     1      | motion_ref_ori_b               |    (6,)    |
        |     2      | base_ang_vel                   |    (3,)    |
        |     3      | joint_pos                      |   (29,)    |
        |     4      | joint_vel                      |   (29,)    |
        |     5      | actions                        |   (29,)    |
        +------------+--------------------------------+------------+
        """
        self.obs = self.obs_manager.compute_group("policy", update_history=True).clamp(-10, 10).numpy()

    def _policy_reasoning(self):

        if cfg.policy_type == "onnx":
            (
                act,
                self.r_joint_pos,
                self.r_joint_vel,
                self.r_body_pos_w,
                self.r_body_quat_w,
                self.r_body_lin_vel_w,
                self.r_body_ang_vel_w,
            ) = self.run_onnx_inference(
                self.policy, self.obs.astype(np.float32), self.time_step
            )
        # print(act.shape)
        # print(self.r_joint_pos.shape)
        # print(self.r_joint_vel.shape)
        # print(self.r_body_pos_w.shape)
        # print(self.r_body_quat_w.shape)
        # print(self.r_body_lin_vel_w.shape)
        # print(self.r_body_ang_vel_w.shape)
        self.action[:] = act.copy()

    def sim_loop(self):
        for i in range(self.control_decimation):
            step_start = time.time()

            if not cfg.motion_play or (cfg.motion_play and cfg.sim_motion_play):
                # tau = self._PD_control()
                tau = self._PD_control(self.target_dof_pos)
                self.d.ctrl[:] = tau
            if not self.paused:
                self.prev_qpos = self.d.qpos.copy()
                self.set_camera()
                # self.d.qpos[0:3] = np.array([0,0,1])
                # self.d.qpos[3:7] = np.array([0,0,0,1])
                # self.d.qvel[0:3] = 0
                # self.d.qvel[3:6] = 0
                mujoco.mj_step(self.m, self.d)
                # self.viewer.sync()
            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def _PD_control(self, _P_t=0):
        P_n = self.d.qpos[7:]
        V_n = self.d.qvel[6:]
        # print(f"P_n:{P_n}")
        KP = self.P_gains
        KD = self.D_gains
        # 在_compute_torques中使用
        t = KP * (_P_t - P_n) - KD * V_n
        # t = KP * (_P_t - P_n) - KD * V_n
        # print(f"KP * (_P_t - P_n):\r\n{KP * (_P_t - P_n)}")
        # print(f" - KD * V_n: \r\n{ - KD * V_n}")
        # print(f"t: \r\n{t}")
        return t

    def contact_force(self):
        force = 0
        for contact_id, contact in enumerate(self.d.contact):
            if contact.efc_address >= 0:  # Valid contact
                forcetorque = np.zeros(6)
                mujoco.mj_contactForce(self.m, self.d, contact_id, forcetorque)
                # print("forcetorque: ",forcetorque)
                force += forcetorque[0]
        self.fz = force / 65 / 9.81
        # print("force: %8.3f"% force)

    def key_callback(self, keycode):
        # 按空格键切换暂停/继续

        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'running'}")
        elif chr(keycode).lower() == "w":
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0
            self.cmd[0] = 0.8
        elif chr(keycode).lower() == "s":
            self.cmd[0] = -0.8
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0

        elif chr(keycode).lower() == "a":
            self.cmd[1] = 0.4
            self.cmd[0] = 0.0
            self.cmd[2] = 0.0
        elif chr(keycode).lower() == "d":
            self.cmd[1] = -0.4
            self.cmd[0] = 0.0
            self.cmd[2] = 0.0
        elif chr(keycode).lower() == "q":
            self.cmd[2] = 1.5
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
        elif chr(keycode).lower() == "e":
            self.cmd[2] = -1.5
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
        # 释放键时重置控制量
        elif keycode == 48:  # keycode=0 表示无按键
            self.cmd[0] = 0.0
            self.cmd[1] = 0.0
            self.cmd[2] = 0.0

    def set_camera(self):
        # self.viewer.cam.distance = 4
        # self.viewer.cam.azimuth = 180  # 135
        # self.viewer.cam.elevation = 0.0
        # self.viewer.cam.fixedcamid = -1
        # self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        # self.viewer.cam.trackbodyid = 1
        ...

    def _init_robot_conf(self):
        self.default_pos = np.array(
            cfg.leg_default_pos
            + cfg.pelvis_default_pos
            + cfg.arm_default_pos,
            dtype=np.float32,
        )  # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.P_gains = np.array(
            cfg.leg_P_gains + cfg.pelvis_P_gains + cfg.arm_P_gains,
        )  # [70.0, 70.0, 3.0, 70.0, 70.0, 70.0, 1.5, 180.0, 180.0, 70.0, 70.0, 180.0, 180.0, 70.0, 70.0, 330.0, 330.0, 20.0, 20.0, 330.0, 330.0, 20.0, 20.0, 70.0, 70.0, 20.0, 20.0, 70.0, 70.0]
        self.D_gains = np.array(
            cfg.leg_D_gains + cfg.pelvis_D_gains + cfg.arm_D_gains,
        )  # [1.5, 1.5, 0.6, 1.5, 1.5, 1.5, 0.3, 2.5, 2.5, 2.0, 2.0, 2.5, 2.5, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5]
        self.tq_max = np.array(
            cfg.leg_tq_max + cfg.pelvis_tq_max + cfg.arm_tq_max,
            dtype=np.float32,
        )
        self.P_n = np.zeros_like(self.default_pos)
        self.V_n = np.zeros_like(self.default_pos)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        mujoco_joint_name = [joint.name for joint in self.spec.joints][1:]
        for i in range(len(mujoco_joint_name)):
            print(
                "  - "
                + mujoco_joint_name[i]
                + ": {kp: "
                + str(self.P_gains[i])
                + ", kd: "
                + str(self.D_gains[i])
                + ", torque_max: "
                + str(self.tq_max[i])
                + ", default_pos: "
                + str(self.default_pos[i])
                + "}"
            )
        print("mujoco_joint_name:\r\n", mujoco_joint_name)
        self.isaac_sim2mujoco_index = [
            cfg.isaac_sim_joint_name.index(name) for name in mujoco_joint_name
        ]
        print("isaac_sim2mujoco_index:\r\n", self.isaac_sim2mujoco_index)
        self.mujoco2isaac_sim_index = [
            mujoco_joint_name.index(name) for name in cfg.isaac_sim_joint_name
        ]
        print("mujoco2isaac_sim_index:\r\n", self.mujoco2isaac_sim_index)
        self.mujoco_all_body_names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.m.nbody)
        ][1:]
        self.mujoco_body_names_indices = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.mujoco_all_body_names
        ]
        print("mujoco_all_body_names:\r\n", self.mujoco_all_body_names)
        self.motion_body_names_in_isaacsim_index = [
            cfg.isaac_sim_link_name.index(name) for name in cfg.motion_body_names
        ]
        print("motion_body_index:\r\n", self.motion_body_names_in_isaacsim_index)
        a = 1

    def _init_policy_conf(self):
        self.body_indexes = torch.tensor(
            self.motion_body_names_in_isaacsim_index,
            dtype=torch.long,
            device="cpu",
        )
        self.motion = MotionLoader(
            cfg.motion_file,
            self.body_indexes,
            "cpu",
        )
        self.policy_dt = cfg.policy_dt
        if cfg.motion_play:
            self.policy_dt = 1 / self.motion.fps
        self.control_decimation = int(self.policy_dt / cfg.simulator_dt)
        print("control_decimation: ", self.control_decimation)
        self.policy = self.load_onnx_model(cfg.policy_path)

        self.h2_action = np.zeros(cfg.action_num, dtype=np.float32)
        self.h_action = np.zeros(cfg.action_num, dtype=np.float32)
        self.action = np.zeros(cfg.action_num, dtype=np.float32)
        self.action_clip = cfg.action_clip

        self.action_scale = cfg.action_scale
        self.action_num = cfg.action_num
        self.obs = np.zeros(cfg.num_single_obs * cfg.frame_stack, dtype=np.float32)
        self.time_step = np.ones(1, dtype=np.float32) * 1
        self.single_obs = np.zeros(cfg.num_single_obs, dtype=np.float32)

    def load_onnx_model(self, onnx_path, device="cpu"):
        providers = (
            ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        )
        session = ort.InferenceSession(onnx_path, providers=providers)
        return session

    def run_onnx_inference(self, session, obs, time_step):
        # 转换为numpy array并确保数据类型正确
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if isinstance(time_step, torch.Tensor):
            time_step = time_step.detach().cpu().numpy()
        # 获取输入名称
        obs_name = session.get_inputs()[0].name
        time_step_name = session.get_inputs()[1].name
        # 运行推理
        (
            actions,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        ) = session.run(
            None,
            {
                obs_name: obs.reshape(1, cfg.num_single_obs),
                time_step_name: time_step.reshape(1, 1),
            },
        )
        # print("outputs shape")
        # print(actions.shape)
        # print(joint_pos.shape)
        # print(joint_vel.shape)
        # print(body_pos_w.shape)
        # print(body_quat_w.shape)
        # print(body_lin_vel_w.shape)
        # print(body_ang_vel_w.shape)
        return (
            actions,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        )  # 默认返回第一个输出

    def init_vel_geom(self, input):
        # create an invisibale geom and add label on it
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_LABEL,
            size=np.array([0.2, 0.2, 0.2]),  # label_size
            pos=self.d.qpos[:3]
            + np.array(
                [0.0, 0.0, 1.0]
            ),  # lebel position, here is 1 meter above the root joint
            mat=np.eye(3).flatten(),  # label orientation, here is no rotation
            rgba=np.array([0, 0, 0, 0]),  # invisible
        )
        geom.label = str(input)  # set label text
        self.viewer.user_scn.ngeom += 1

    def update_vel_geom(self):
        # update the geom position and label text
        geom = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1]
        geom.pos = self.d.qpos[:3] + np.array([0.0, 0.0, 1.0])
        geom.label = "rb h{:.2f} \r\nGoal Vel: x: {:.2f}, y: {:.2f}, yaw: {:.2f},force_z: {:.2f}".format(
            # self.data["robot.data.body_pos_w"].detach().cpu().numpy()[0][2],
            0.0,
            self.cmd[0],
            self.cmd[1],
            self.cmd[2],
            self.fz,
        )
