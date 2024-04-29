from dataset.yarr.utils.observation_type import ObservationElement
from dataset.yarr.replay_buffer.replay_buffer import ReplayBuffer, ReplayElement

from dataset.yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
import numpy as np

def create_replay(batch_size: int,
                  timesteps: int,
                  save_dir: str,
                  cameras: list,
                  voxel_sizes,
                  replay_size=3e5):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512
    IMAGE_SIZE1 = 128
    IMAGE_SIZE2 = 128

    # low_dim_state
    observation_elements = []    
    observation_elements.append(
        ObservationElement('prev_gripper_pose', (gripper_pose_size,), np.float32))
        
    
    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:        
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, IMAGE_SIZE1, IMAGE_SIZE2,), np.float32))
        # observation_elements.append(
        #     ObservationElement('%s_depth' % cname, (1, IMAGE_SIZE, IMAGE_SIZE,), np.float32))
        # observation_elements.append(
        #     ObservationElement('%s_point_cloud' % cname, (3, IMAGE_SIZE, IMAGE_SIZE,), np.float32)) # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        # observation_elements.append(
        #     ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        # observation_elements.append(
        #     ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size,),
                      np.float32),
        ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
                      np.int32),
        ReplayElement('ignore_collisions', (ignore_collisions_size,),
                      np.int32),
        ReplayElement('gripper_pose', (gripper_pose_size,),
                      np.float32),        
        ReplayElement('gripper_rot', (trans_indicies_size,),
                      np.float32),                              
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), bool),
    ]

    replay_buffer = UniformReplayBuffer( # all tuples in the buffer have equal sample weighting
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(7,), # 3 translation + 3 rotation + 1 gripper open
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer