import torch
import os
import cv2 as cv
import numpy as np
import torchvision
import albumentations as A
import random
import einops
from utilities import augmentations
from tqdm import tqdm
import matplotlib.pyplot as plt
from annotation_utils import read_annotation, get_insar_path

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.data_path = config["data_path"]
        self.augmentations = augmentations.get_augmentations(config)
        self.interferograms = []
        self.channels = config['num_channels']
        frames = os.listdir(self.data_path)
        for frame in tqdm(frames):
            frame_path = self.data_path + '/'+frame +'/interferograms/'
            caption = os.listdir(frame_path)
            for cap in caption:
                caption_path = frame_path + cap +'/'+cap+ '.geo.diff.png'

                if os.path.exists(caption_path):
                    image_dict = {'path': caption_path, 'frame': frame}
                    self.interferograms.append(image_dict)

        self.num_examples = len(self.interferograms)


    def __len__(self):
        return self.num_examples


    def prepare_insar(self,insar):
        insar = torch.from_numpy(insar).float().permute(2,0,1)
        insar /= 255
        return insar

    def read_insar(self,path):
        insar = cv.imread(path,0)
        if insar is None:
            print('None')
            return insar

        insar = einops.repeat(insar, 'h w -> h w c', c=self.channels)
        transform = self.augmentations(image=insar)
        insar_1 = transform['image']
        transform_2 = self.augmentations(image=insar)
        insar_2 = transform_2['image']

        insar_1 = self.prepare_insar(insar_1)
        insar_2 = self.prepare_insar(insar_2)
        return (insar_1,insar_2)

    def __getitem__(self, index):
        insar = None
        while insar is None:
            sample = self.interferograms[index]
            path = sample['path']

            insar = self.read_insar(path)
            if insar is None:
                if index<self.num_examples-1:
                    index +=1
                else:
                    index = 0


        return insar,0
