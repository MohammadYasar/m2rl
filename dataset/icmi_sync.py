import os
import argparse
import cv2
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from frame_match import *
import datetime
import multiprocessing
import glob
from natsort import natsorted
from PIL import Image
import threading


def make_video(file_path, corr_indices, cam_identifier):
    count = 0
    # Path for writing new images 
    write_folder = file_path + "synchronized" + cam_identifier
    os.makedirs(write_folder, exist_ok=True)

    # Path for reading all unsynchronized images 
    file_path = file_path + cam_identifier
    camera = file_path.split("/")[-1]
    # output_video_file = file_path.replace(camera, "synchronized_{}.mp4".format(camera))

    image_path = file_path + "/*.png"
    # print ("image path ", image_path, camera)
    # first_image_path = natsorted(glob.glob(image_path))[0]
    # first_image = cv2.imread(first_image_path)
    # fps = 15
    # width = int(first_image.shape[1])
    # height = int(first_image.shape[0])
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (codec options may vary)
    # video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))


    for i, image_path in enumerate(natsorted(glob.glob(image_path))):
        
        if i in corr_indices:
            # print ("image_path ", count)
            cv2_image = cv2.imread(image_path)
            pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
            pil_image.save(f'{write_folder}/{count}.png')


            # cv2.imwrite()
            # video_writer.write(frame)
        
            count +=1

    # cv2.destroyAllWindows()
    # video_writer.release()
    # print ("writing video in ", output_video_file)
    return
    
def make_only_video(file_path="/project/CollabRoboGroup/datasets/icmi_cam_data/P_128/Task_4/Interface_1/Trial_1synchronized/", cam_identifier='kinect1_color'):
    ount = 0
    # Path for writing new images 
    write_folder = file_path + cam_identifier
    os.makedirs(write_folder, exist_ok=True)

    # Path for reading all unsynchronized images 
    file_path = file_path + cam_identifier
    print ("file_path ", file_path)
    camera = file_path.split("/")[-1]
    output_video_file = file_path.replace(camera, "synchronized_{}.mp4".format(camera))

    image_path = file_path + "/*.png"
    first_image_path = natsorted(glob.glob(image_path))[0]

    first_image = cv2.imread(first_image_path)
    fps = 15
    width = int(first_image.shape[1])
    height = int(first_image.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (codec options may vary)
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))


    for i, image_path in enumerate(natsorted(glob.glob(image_path))):
        # if i in corr_indices:
        # print ("image_path ", count)
        cv2_image = cv2.imread(image_path)
        # pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        # pil_image.save(f'{write_folder}/{count}.png')

        # cv2.imwrite()
        video_writer.write(cv2_image)
    
    video_writer.release()


    return

def get_fileName(_trial):
    robot_pose_dir = _trial.replace("icmi_cam_data", "icmi_cam_data/input_and_pose_data") 
    print ("robot_pose_dir ", robot_pose_dir)
    robot_pose_path = glob.glob(robot_pose_dir + "/*.csv")[0]                
    realsense_path = _trial + "/realsense_timestamps.pkl"
    kinect1_path = _trial + "/kinect1_timestamps.pkl"
    kinect2_path = _trial + "/kinect2_timestamps.pkl"
    
    return robot_pose_path, realsense_path, kinect1_path, kinect2_path


def write_skeleton(corr_indices, robot_pose_data, robot_pose_path, file_path):
    print (len(corr_indices))
    write_folder = file_path + "synchronized" + "/robot_data.csv"
    
    new_robot_pose = list()
    df = pd.read_csv(robot_pose_path)
    print ("old df ",len(df))
    df = df.iloc[:len(corr_indices)]

    df = df.reset_index(drop=True)
    print ("new df ",len(df))

    # for i in range(0, len(corr_indices)):
    #     new_robot_pose.append(robot_pose_data[corr_indices[i]])
    # new_robot_pose = np.asarray(new_robot_pose)
    # new_robot_pose = pd.DataFrame(new_robot_pose)
    # print ("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/processed_skeleton.csv".format(experiment_type,variation_num,group_num,trial))
    df.to_csv(write_folder)

def read_pickle(pickle_file):
    with open(pickle_file, 'rb') as _file:
        data = pickle.load(_file)     

    return np.asarray(data).reshape(-1,1)


def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    data = np.asarray(data)
    return (data)[1:, 0].reshape(-1,1), data

count = 0    
data_dir = "/project/CollabRoboGroup/datasets/icmi_cam_data/P_*"

# P_*/Task_*/Interface_*/Trial_* 

make_only_video(cam_identifier='kinect1_color')
for _dir in sorted(glob.glob(data_dir), reverse=True):
    task_dir = _dir + "/Task_*"
    for _task in sorted(glob.glob(task_dir)):
        interface_dir = _task + "/Interface_*"
        for _interface in sorted(glob.glob(interface_dir)):
            trial_dir = _interface + "/Trial_*" 
            for _trial in sorted(glob.glob(trial_dir)):
                # Assigning path variables
                if "synchronized" in _trial:
                    continue
                
                print ("trial_dir ", _trial)
                robot_pose_path, realsense_path, kinect1_path, kinect2_path = get_fileName(_trial)
                
                if os.path.exists(realsense_path) and os.path.exists(kinect1_path) and os.path.exists(kinect2_path) and os.path.exists(robot_pose_path):
                    print ("reading pickle files")
                    realsense_timestamps = read_pickle(realsense_path)
                    kinect1_timestamps = read_pickle(kinect1_path)
                    kinect2_timestamps = read_pickle(kinect2_path)
                    robot_timestamps, robot_pose_data = read_csv(robot_pose_path)
                    query_timestamp = robot_timestamps
                    if len(robot_timestamps)>1:
                        query_timestamp, corr_indices_k1_k1 = loop_queryarray(query_timestamp, kinect1_timestamps, denom=1)
                        query_timestamp, corr_indices_k1_k2 = loop_queryarray(query_timestamp, kinect2_timestamps, denom=1)
                        _, corr_indices_k1_w = loop_queryarray(query_timestamp, realsense_timestamps, denom=1)
                        _, corr_indices_k1_r = loop_queryarray(query_timestamp, robot_timestamps, denom=1)
                        print (corr_indices_k1_k2[:10], corr_indices_k1_r[:10], corr_indices_k1_w[:10])
                        print ("timestamps for all the streans ", len(set(corr_indices_k1_w)),len(set(corr_indices_k1_k2)),len(set(corr_indices_k1_r)), len(set(corr_indices_k1_k1)))
                        
                        # print ("writing ", corr_indices_k1_k2[:])
                    
                        thread1 = threading.Thread(target=make_video, args =(_trial, corr_indices_k1_w, "/rs_color"))
                        thread2 = threading.Thread(target=make_video, args =(_trial, corr_indices_k1_k1, "/kinect1_color"))
                        thread3 = threading.Thread(target=make_video, args =(_trial, corr_indices_k1_k2, "/kinect2_color"))
                        thread1.start()
                        thread2.start()
                        thread3.start()

                        # Wait for all threads to finish
                        thread1.join()
                        thread2.join()
                        thread3.join()
                        write_skeleton(list(set(corr_indices_k1_r)), robot_pose_data, robot_pose_path, _trial)                    

                        # make_video(_trial, corr_indices_k1_w, "/rs_color")
                        # make_video(_trial, corr_indices_k1_k1, "/kinect1_color")
                        # make_video(_trial, corr_indices_k1_k2, "/kinect2_color")
                else:
                    print ("missing pickle files")
                
print ("count ", count)                    
