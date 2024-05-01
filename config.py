from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function



import os.path as osp
import sys
import time
import numpy as np
import argparse
from easydict import EasyDict as edict


C = edict()
config = C
cfg = C

C.seed = 888

C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'bc_agent'


C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log'))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.this_dir, 'models'))

"""Logging Config"""
C.wandb_activate = True
C.data_dir = "/project/CollabRoboGroup/datasets/franka_multimodal_teleop/"
C.train_split = 12
C.curr_interface = 2
C.task_id = '3'

"""Scene Config"""
C.SCENE_BOUNDS = [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]
C.CAMERAS = ['wrist', 'left_shoulder', 'right_shoulder']
C.BATCH_SIZE = 4
C.VOXEL_SIZES = [100]
C.IMAGE_SIZE = 128
C.device = 'cuda'

C.LOG_FREQ = 1
C.TRAINING_ITERATIONS = 10000
"""Model Config"""
C.agent = 'diffuser_agent'
C.optimizer = 'lamb'
C.lr = 1e-4
C.initial_dim=512,
C.low_dim_size=64
