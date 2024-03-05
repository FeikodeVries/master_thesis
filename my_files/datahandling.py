import pathlib
import os
import numpy as np
import torch


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


