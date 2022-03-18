from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import cv2
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

class ImslpDataset(Dataset):
    """IMSLP dataset."""

    def __init__(self, split_file, root_dir, transform=None):
        """
        Args:
            split_file (string): Path to the train-test split file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        my_file = open(split_file, "r")
        self.split = my_file.readlines() #Contains an \n at the end. Remove it while calling it
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.split[idx][:-1])

        image = io.imread(img_name)
        image = cv2.resize(image, (1024, 1024))
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

