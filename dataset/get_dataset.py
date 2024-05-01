import os 
import numpy as numpy
import pandas as pd
import numpy as np
from dataset.demo import Demo
from dataset.observation import Observation
import glob
from dataset.utils import keypoint_discovery, read_robot_pose, _add_keypoints_to_replay
from dataset.replay_buffer import create_replay
from dataset.yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer

# cameras = ['kinect_left']
cameras = ['wrist', 'left_shoulder', 'right_shoulder']

def get_low_obs(obs_file):
    obs = pd.read_csv(obs_file)
    obs = np.asarray(obs)


def get_demo(replay, robot_obs_dir, robot_data_file, interface_id="spacemouse"):
    episode_keypoints, demo = read_robot_pose(robot_obs_dir, robot_data_file, interface_id)    
    # fill_buffer(demo)
    
    voxel_sizes = [100]
    episode_keypoints = episode_keypoints[0]
    rotation_resolution = 5
    for i in range(len(demo) - 1):
        
        obs = demo[i]
        initial_pose = demo[i].gripper_pose
        # np.concatenate([demo[i].gripper_pose, demo[i].gripper_rot, demo[i].gripper_open]) # for the diffusion model
        
        # if our starting point is past one of the keypoints, then remove it
        while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
            episode_keypoints = episode_keypoints[1:]
        if len(episode_keypoints) == 0:
            break

        _add_keypoints_to_replay(
            replay, obs, initial_pose, demo, episode_keypoints, cameras,
            voxel_sizes,
            rotation_resolution, crop_augmentation=False, device='cuda')

train_replay_buffer = create_replay(batch_size=4,
                                    timesteps=1,
                                    save_dir='/scratch/msy9an/replay_train',
                                    cameras=cameras,
                                    voxel_sizes=[100])


test_replay_buffer = create_replay(batch_size=4,
                                    timesteps=1,
                                    save_dir='/scratch/msy9an/replay_test',
                                    cameras=cameras,
                                    voxel_sizes=[100])                
