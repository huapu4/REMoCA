# -*- coding: utf-8 -*-
"""
@Time    : 2021/10/28
@Author  : Lin Zhenzhe, Zhang Shuyi
"""

import os.path

import torch
import os


class Defalut_parameters(object):
    # calculate uints
    def __init__(self):
        self.set_device = 0
        self.devices_ids = [0, 1, 2, 3]
        self.task = ['1_FT', '2_ST', '3_AST', '4_HSPT', '5_VSPT']
        self.data_dir = './data/'
        # self.exter_task = [''1_FT', '2_ST', '3_AST', '4_HSPT', '5_VSPT']
        # self.exter_data_dir = ['./data/exter_val/huiai_data', './data/exter_val/eryuan_data',
        #                       './data/exter_val/zoc318_data', './data/exter_val/all_data'][-1]
        self.model_dir = './model'
        self.log_dir = './check_log'
        self.nEpochs = 1000  # Number of epochs for training
        self.resume_epoch = 0  # Default is 0, change if want to resume
        self.useTest = True  # See evolution of the test set when training
        self.nTestInterval = 20  # Run on test set every nTestInterval epochs
        self.snapshot = 10  # Store a model every snapshot epochs
        self.lr = 5e-4  # Learning rate
        self.prepare = {'train': True, 'test': False}  # prepare train/val set
        self.clip_len = 20
        self.batch_size = 100

    def get_task_dir(self, run_task):
        self.run_task = run_task
        set_dict = {}
        for t in self.task:
            set_dict[t] = {'data_loadpath': os.path.join(self.data_dir, t),
                           'model_savepath': os.path.join(self.model_dir, t),
                           'check_log': os.path.join(self.log_dir, t)}
        return set_dict[self.run_task]

    def get_exter_task_dir(self, run_task):
        self.run_task = run_task
        set_dict = {}
        for t in self.task:
            set_dict[t] = {'data_loadpath': os.path.join(self.exter_data_dir, t),
                           'model_savepath': os.path.join(self.model_dir, t),
                           'check_log': os.path.join(self.log_dir, t)}
        return set_dict[self.run_task]