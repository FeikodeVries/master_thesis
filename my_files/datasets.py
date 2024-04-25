"""
Methods to implement the classes
"""
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
import os
import json
import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm
import pathlib


class BaseDataset(data.Dataset):
    """
    Load the data for the VAE to train on
    """

    # In PinballDataset, they provide numbers indicating maximum value after the ordereddict, should i do this?

    VAR_INFO = OrderedDict({
        'base_x': 'continuous',
        'base_y': 'continuous',
        'base_vel_x': 'continuous',
        'base_vel_y': 'continuous',
        'tip_x': 'continuous',
        'tip_y': 'continuous',
        'tip_vel_x': 'continuous',
        'tip_vel_y': 'continuous'
    })

    CAUSAL_VAR_NAMES = ['base', 'root']

    def __init__(self, data_folder, interventions, split='train', single_image=False, seq_len=2, **kwargs):
        super().__init__()
        image_path = data_folder + 'images.npz'
        # target_path = data_folder + 'intervention.npz'
        self.imgs = torch.from_numpy(np.load(image_path)['entries'])
        # self.targets = torch.from_numpy(np.load(target_path)['entries'])
        self.targets = interventions

        self._clean_up_data()
        self.split_name = split
        if split.startswith('val'):
            self.split_name = self.split_name.replace('val', 'test')
        self.single_image = single_image
        self.encodings_active = False
        self.seq_len = seq_len if not single_image else 1

    def _clean_up_data(self):
        """
        Push the channels to the PyTorch dimension if needed
        :return:
        """
        if len(self.imgs.shape) == 5:
            self.imgs = self.imgs.permute(0, 1, 4, 2, 3)  # Push channels to PyTorch dimension
        else:
            self.imgs = self.imgs.permute(0, 3, 1, 2)

        self.target_names = ReacherDataset.CAUSAL_VAR_NAMES
        # print(f'Using the causal variables {self.target_names}')

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            return imgs

    def load_encodings(self, filename):
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def label_to_img(self, label):
        return (label + 1.0) / 2.0

    def num_labels(self):
        return -1

    def get_imgs(self, idx):
        return self.imgs[idx]

    def num_vars(self):
        return self.targets.shape[-1]

    def target_names(self):
        return self.target_names

    def get_img_width(self):
        return self.imgs.shape[-2]

    def get_inp_channels(self):
        return self.imgs.shape[-3]

    def get_causal_var_info(self):
        return ReacherDataset.VAR_INFO

    def __len__(self):
        return self.imgs.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        returns = []

        img_pair = self.imgs[idx:idx + self.seq_len]
        target = self.targets[idx:idx + self.seq_len - 1]

        if self.single_image:
            img_pair = img_pair[0]
        else:
            returns += [target]

        img_pair = self._prepare_imgs(img_pair)
        returns = [img_pair] + returns + [idx]

        return tuple(returns) if len(returns) > 1 else returns[0]


class ReacherDataset(BaseDataset):
    VAR_INFO = OrderedDict({
        'base_x': 'continuous',
        'base_y': 'continuous',
        'base_vel_x': 'continuous',
        'base_vel_y': 'continuous',
        'tip_x': 'continuous',
        'tip_y': 'continuous',
        'tip_vel_x': 'continuous',
        'tip_vel_y': 'continuous'
    })

    CAUSAL_VAR_NAMES = ['base', 'root']


class WalkerDataset(BaseDataset):
    VAR_INFO = OrderedDict({
        'right_foot_x': 'continuous',
        'right_foot_y': 'continuous',
        'right_foot_vel_x': 'continuous',
        'right_foot_vel_y': 'continuous',
        'left_foot_x': 'continuous',
        'left_foot_y': 'continuous',
        'left_foot_vel_x': 'continuous',
        'left_foot_vel_y': 'continuous',
        'right_low_leg_x': 'continuous',
        'right_low_leg_y': 'continuous',
        'right_low_leg_vel_x': 'continuous',
        'right_low_leg_vel_y': 'continuous',
        'left_low_leg_x': 'continuous',
        'left_low_leg_y': 'continuous',
        'left_low_leg_vel_x': 'continuous',
        'left_low_leg_vel_y': 'continuous',
        'right_upper_leg_x': 'continuous',
        'right_upper_leg_y': 'continuous',
        'right_upper_leg_vel_x': 'continuous',
        'right_upper_leg_vel_y': 'continuous',
        'left_upper_leg_x': 'continuous',
        'left_upper_leg_y': 'continuous',
        'left_upper_leg_vel_x': 'continuous',
        'left_upper_leg_vel_y': 'continuous'
    })

    CAUSAL_VAR_NAMES = ['right_upper_leg', 'right_lower_leg', 'right_foot',
                        'left_upper_leg', 'left_lower_leg', 'left_foot']
