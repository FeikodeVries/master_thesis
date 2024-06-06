import pathlib
import os
import numpy as np
import torch

import torch.utils.data as data
from cleanrl.my_files.datasets import ReacherDataset, WalkerDataset, PendulumDataset, CheetahDataset


class DataHandling:
    """
    Load, save, and augment numpy array files
    """
    def __init__(self, folder: str = '/data/'):
        self.path = str(pathlib.Path(__file__).parent.resolve()) + folder

    def update_npz_file(self, data, filename: str = 'images', batch_size: int = 2):
        """
        Load numpy array file, and create one if none exists. Then add the new image to the existing dataset
        Update the dataset to contain only the given amount of images
        :param data: Matrix representing an image
        :param filename: Name of the file to save to
        :param batch_size: How many of the observations to save in the file
        """
        batch_size = batch_size - 1
        filepath = self.path + (filename+'.npz')
        if filename == 'current':
            np.savez_compressed(filepath, entries=np.array([data]))
        else:
            if os.path.isfile(filepath):
                data_set = np.load(filepath)
                if batch_size > len(data_set['entries']):
                    updated_data = np.concatenate((data_set['entries'], np.array([data])), axis=0)
                else:
                    updated_data = np.concatenate((data_set['entries'][-batch_size:], np.array([data])), axis=0)
                np.savez_compressed(filepath, entries=updated_data)
            else:
                np.savez_compressed(filepath, entries=np.array([data]))

        return filepath

    def batch_update_npz(self, data, filename: str = 'images'):
        filepath = self.path + (filename + '.npz')
        np.savez_compressed(filepath, entries=data)
        return filepath


# def load_datasets(args, env_name):
#     pl.seed_everything(args.seed)
#     print('Loading data...')
#
#     # Extend for different models
#     if env_name == 'Reacher':
#         DataClass = ReacherDataset
#         dataset_args = {}
#         test_args = lambda train_set: {'causal_vars': train_set.target_names}
#     elif env_name == 'Walker':
#         DataClass = WalkerDataset
#         dataset_args = {}
#         test_args = lambda train_set: {'causal_vars': train_set.target_names}
#     else:
#         pass
#     folder = str(pathlib.Path(__file__).parent.resolve()) + '/data/'
#
#     train_data = DataClass(data_folder=folder, split='train', single_image=False, seq_len=2, **dataset_args)
#     # val_data = DataClass(data_folder=folder, split='val_indep', single_image=True, **dataset_args, **test_args(train_data))
#     train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
#                                    pin_memory=True, drop_last=True, num_workers=args.num_workers)
#
#     print(f'Training dataset size: {len(train_data)} / {len(train_loader)}')
#
#     datasets = {
#         'train': train_data,
#         # 'val': val_data
#     }
#     data_loaders = {
#         'train': train_loader
#     }
#
#     return datasets, data_loaders, env_name.lower()


def load_data_new(args, img_data, interventions, env_name, seq_len=3):
    print('Loading data...')
    env_name = env_name.split('-')[0]
    DataClass = WalkerDataset
    dataset_args = {}
    test_args = lambda train_set: {'causal_vars': train_set.target_names}
    # Extend for different models
    # if env_name == 'Reacher':
    #     DataClass = ReacherDataset
    #     dataset_args = {}
    #     test_args = lambda train_set: {'causal_vars': train_set.target_names}
    # elif env_name == 'Walker2d':
    #     DataClass = WalkerDataset
    #     dataset_args = {}
    #     test_args = lambda train_set: {'causal_vars': train_set.target_names}
    # elif env_name == 'InvertedPendulum':
    #     # TODO: Sizes of tensors not matching?
    #     DataClass = PendulumDataset
    #     dataset_args = {}
    #     test_args = lambda train_set: {'causal_vars': train_set.target_names}
    # elif env_name == 'HalfCheetah':
    #     # TODO: far less often doing print statements for performance??
    #     DataClass = CheetahDataset
    #     dataset_args = {}
    #     test_args = lambda train_set: {'causal_vars': train_set.target_names}
    # else:
    #     pass
    folder = str(pathlib.Path(__file__).parent.resolve()) + '/data/'

    train_data = DataClass(data_folder=folder, img_data=img_data, interventions=interventions,
                           split='train', single_image=False, seq_len=seq_len, **dataset_args)
    # TODO: pin_memory and num_workers had to be disabled to allow for the use of dataloaders strangely
    train_loader = data.DataLoader(train_data, batch_size=args.minibatch_size, shuffle=True,
                                   pin_memory=False, drop_last=True, num_workers=0)

    datasets = {
        'train': train_data
    }
    data_loaders = {
        'train': train_loader
    }

    return datasets, data_loaders, env_name.lower()

