import numpy as np
data_path = "/home/unitree/unitree_rl_gym-main/joint_state.npz"
action_path = "/home/unitree/unitree_rl_gym-main/action.npz"
data = np.load(data_path, allow_pickle=True)
actions = np.load(action_path,allow_pickle=True)
qpos = data['joint_positions']
qvel = data['joint_velocities']
tau = data['joint_efforts']
action = actions['action']
import pdb;pdb.set_trace()