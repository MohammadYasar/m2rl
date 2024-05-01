import os 
import glob
import time
import torch
from dataset.get_dataset import *
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from config import config
import wandb
from agents.diffuser_agent import *
from agents.bc_agent.agent import *

run_name = "{}_{}_interface_{}".format(config.agent, config.task_id, config.curr_interface)
if config.wandb_activate == True:
    wandb.init(project="diffuser_actor", name=run_name, entity="crl_rl2", dir="/scratch/msy9an/icmi_log/")
    wandb.config.update(config)
    

def get_train_dataset():
    for _dir in sorted(glob.glob(config.data_dir), reverse=True):
        i = 1
        task_dir = _dir + "/task_{}".format(config.task_id)
        for _task in sorted(glob.glob(task_dir)):
            interface_dir = _task + "/interface_{}".format(config.curr_interface)
            for _interface in range(1, config.curr_interface+1): #sorted(glob.glob(interface_dir)):
                interface_dir = _task + "/interface_{}".format(_interface)
                for epx_id in range(config.train_split):
                    trial_dir = interface_dir + "/episode_{}_synchronized".format(epx_id) 
                    for _trial in sorted(glob.glob(trial_dir)):
                        # Assigning path variables
                        if not "synchronized" in _trial:
                            continue
        
                        robot_obs_dir = _trial
                        robot_pose_path = glob.glob(robot_obs_dir + "/*.csv")[0] 

                        interface_id = "joystick" if interface_dir.split("/")[-1] ==  "interface_2" else "spacemouse"
                        get_demo(train_replay_buffer, robot_obs_dir, robot_pose_path, interface_id)               

    train_wrapped_replay = PyTorchReplayBuffer(train_replay_buffer)
    train_dataset = train_wrapped_replay.dataset()
    return train_dataset



def get_test_dataset():
    for _dir in sorted(glob.glob(config.data_dir), reverse=True):
        i = 1
        # task_dir = _dir + "/task_{}".format(i)
        task_dir = os.path.join(_dir, "task_{}".format(config.task_id))
        for _task in sorted(glob.glob(task_dir)):
            interface_dir = _task + "/interface_{}".format(config.curr_interface)
            for _interface in range(1, config.curr_interface+1): #sorted(glob.glob(interface_dir)):
                interface_dir = _task + "/interface_{}".format(_interface)
                for epx_id in range(config.train_split, 16):
                    trial_dir = interface_dir + "/episode_{}_synchronized".format(epx_id) 
                    # print ("trial_dir ", trial_dir, _interface)
                    for _trial in sorted(glob.glob(trial_dir)):
                        # Assigning path variables
                        print ("test_trial ", _trial)
                        if not "synchronized" in _trial:
                            continue
        
                        robot_obs_dir = _trial
                        robot_pose_path = glob.glob(robot_obs_dir + "/*.csv")[0] 

                        interface_id = "joystick" if interface_dir.split("/")[-1] ==  "interface_2" else "spacemouse"
                        get_demo(test_replay_buffer, robot_obs_dir, robot_pose_path, interface_id)               
    test_wrapped_replay = PyTorchReplayBuffer(test_replay_buffer)
    test_dataset = test_wrapped_replay.dataset()
    return test_dataset

def create_agent():
    # initialize PerceiverActor
    
    if config.agent == 'diffuser_agent':
        mvdit = MoveDit(depth=1, 
                    iterations=1,
                    voxel_size=config.VOXEL_SIZES[0],
                    initial_dim=config.initial_dim,
                    low_dim_size=config.low_dim_size,
                    )

        agent = DiffuserActorAgent(
            coordinate_bounds=config.SCENE_BOUNDS,
            perceiver_encoder=mvdit,
            camera_names=config.CAMERAS,
            batch_size=config.BATCH_SIZE,
            voxel_size=config.VOXEL_SIZES[0],
            voxel_feature_size=3,
            num_rotation_classes=72,
            rotation_resolution=5,
            lr=config.lr,
            image_resolution=[config.IMAGE_SIZE, config.IMAGE_SIZE],
            lambda_weight_l2=0.000001,
            transform_augmentation=False,
            optimizer_type=config.optimizer,
        )
        agent.build(training=True, device=config.device)
    elif config.agent == 'bc_agent':
        mvdit = MoveDit(depth=1, 
                    iterations=1,
                    voxel_size=config.VOXEL_SIZES[0],
                    initial_dim=config.initial_dim,
                    low_dim_size=config.low_dim_size,
                    obs_dim = 512*2)

        agent = BCAgent(
            coordinate_bounds=config.SCENE_BOUNDS,
            perceiver_encoder=mvdit,
            camera_names=config.CAMERAS,
            batch_size=config.BATCH_SIZE,
            voxel_size=config.VOXEL_SIZES[0],
            voxel_feature_size=3,
            num_rotation_classes=72,
            rotation_resolution=5,
            lr=config.lr,
            image_resolution=[config.IMAGE_SIZE, config.IMAGE_SIZE],
            lambda_weight_l2=0.000001,
            transform_augmentation=False,
            optimizer_type=config.optimizer,
        )
        agent.build(training=True, device=config.device)

    return agent


def run_experiment():
    test_dataset = get_test_dataset()
    train_dataset = get_train_dataset()
    diff_agent = create_agent()
    
    train_data_iter = iter(train_dataset)
    test_data_iter = iter(test_dataset)

    loss_list = list()
    start_time = time.time()

    for iteration in range(config.TRAINING_ITERATIONS):
        batch = next(train_data_iter)
        test_batch = next(test_data_iter)

        batch = {k: v.to(config.device) for k, v in batch.items() if type(v) == torch.Tensor}
        test_batch = {k: v.to(config.device) for k, v in test_batch.items() if type(v) == torch.Tensor}

        update_dict = diff_agent.update(iteration, batch)
        eval_dict = diff_agent.evaluate(iteration, test_batch)
        loss_list.append(update_dict['total_loss'])
        if iteration % config.LOG_FREQ == 0:
            elapsed_time = (time.time() - start_time) / 60.0
            # print("Total Training Loss: %f | Elapsed Time: %f mins" % (update_dict['total_loss'], elapsed_time))
            if config.wandb_activate == True:
                wandb.log({"iteration": iteration, "train_loss/total": update_dict['total_loss'], "train_loss/translation": update_dict['trans_loss'], 
                "train_loss/rotation": update_dict['rot_loss'], "train_loss/gripper": update_dict['grip_loss'],
                "test_loss/total": eval_dict['total_loss'], "test_loss/translation": eval_dict['trans_loss'], 
                "test_loss/rotation": eval_dict['rot_loss'], "test_loss/gripper": eval_dict['grip_loss']            
                })

            # print("Total Test Loss: %f | Elapsed Time: %f mins" % (eval_dict['total_loss'], elapsed_time))




run_experiment()




