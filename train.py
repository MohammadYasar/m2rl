import os 
import glob
import time
import torch
from dataset.get_dataset import *
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from config import *
from agents.diffuser_agent import *


def get_train_dataset():
    for _dir in sorted(glob.glob(data_dir)):
        task_dir = _dir + "/Task_*"
        for _task in sorted(glob.glob(task_dir)):
            interface_dir = _task + "/Interface_*"
            for _interface in sorted(glob.glob(interface_dir)):
                trial_dir = _interface + "/Trial_*" 
                for _trial in sorted(glob.glob(trial_dir)):
                    print ("trial ", _trial)
                    if not "synchronized" in _trial:
                        continue
                    # Assigning path variables
                    robot_obs_dir = _trial
                    robot_pose_path = glob.glob(robot_obs_dir + "/*.csv")[0] 
                    print ("interface_id ", interface_dir)

                    interface_id = "joystick" if _interface.split("/")[-1] ==  "Interface_2" else "spacemouse"
                    get_demo(train_replay_buffer, robot_obs_dir, robot_pose_path, interface_id)               
            break

    train_wrapped_replay = PyTorchReplayBuffer(train_replay_buffer)
    train_dataset = train_wrapped_replay.dataset()
    return train_dataset

def create_agent():
    # initialize PerceiverActor
    mvdit = MoveDit(depth=1, 
                    iterations=1,
                    voxel_size=100,
                    initial_dim=512,
                    low_dim_size=64)

    diff_agent = DiffuserActorAgent(
        coordinate_bounds=SCENE_BOUNDS,
        perceiver_encoder=mvdit,
        camera_names=CAMERAS,
        batch_size=BATCH_SIZE,
        voxel_size=VOXEL_SIZES[0],
        voxel_feature_size=3,
        num_rotation_classes=72,
        rotation_resolution=5,
        lr=0.00001,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        lambda_weight_l2=0.000001,
        transform_augmentation=False,
        optimizer_type='lamb',
    )
    diff_agent.build(training=True, device=device)
    return diff_agent


def run_experiment():
    train_dataset = get_train_dataset()
    diff_agent = create_agent()
    train_data_iter = iter(train_dataset)
    loss_list = list()
    start_time = time.time()
    for iteration in range(TRAINING_ITERATIONS):
        batch = next(train_data_iter)
        batch = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}
        update_dict = diff_agent.update(iteration, batch)
        loss_list.append(update_dict['total_loss'])
        if iteration % LOG_FREQ == 0:
            elapsed_time = (time.time() - start_time) / 60.0
            print("Total Loss: %f | Elapsed Time: %f mins" % (update_dict['total_loss'], elapsed_time))




run_experiment()




