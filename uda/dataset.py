from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from imslp import ImslpDataset
# Ignore warnings
import warnings
from transforms_imslp import *

warnings.filterwarnings("ignore")


data_transform = transforms.Compose([
        ToTensor()
    ])

dataset = ImslpDataset(split_file='../../datasets/imslp_dataset/train_test_split/test_list.txt', root_dir='../../datasets/imslp_dataset/images/', transform=data_transform)
fig = plt.figure()

for i in range(len(dataset)):
    sample = dataset[i]

    #print(i, sample['image'].shape)

dataset_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1, shuffle=True,
                                             num_workers=4)