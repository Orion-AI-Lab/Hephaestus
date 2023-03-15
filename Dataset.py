import os

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm
import json
from utilities import augmentations
import random
import einops

np.random.seed(999)
torch.manual_seed(999)
random.seed(999)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.data_path = config["data_path"]
        self.augmentations = augmentations.get_augmentations(config)
        self.interferograms = []
        self.channels = config["num_channels"]
        frames = os.listdir(self.data_path)
        for frame in tqdm(frames):
            frame_path = self.data_path + "/" + frame + "/interferograms/"
            caption = os.listdir(frame_path)
            for cap in caption:
                caption_path = frame_path + cap + "/" + cap + ".geo.diff.png"

                if os.path.exists(caption_path):
                    image_dict = {"path": caption_path, "frame": frame}
                    self.interferograms.append(image_dict)

        self.num_examples = len(self.interferograms)

    def __len__(self):
        return self.num_examples

    def prepare_insar(self, insar):
        insar = torch.from_numpy(insar).float().permute(2, 0, 1)
        insar /= 255
        return insar

    def read_insar(self, path):
        insar = cv.imread(path, 0)
        if insar is None:
            print("None")
            return insar

        insar = np.expand_dims(insar, axis=2).repeat(self.channels, axis=2)
        transform = self.augmentations(image=insar)
        insar_1 = transform["image"]
        transform_2 = self.augmentations(image=insar)
        insar_2 = transform_2["image"]

        insar_1 = self.prepare_insar(insar_1)
        insar_2 = self.prepare_insar(insar_2)
        return (insar_1, insar_2)

    def __getitem__(self, index):
        insar = None
        while insar is None:
            sample = self.interferograms[index]
            path = sample["path"]

            insar = self.read_insar(path)
            if insar is None:
                if index < self.num_examples - 1:
                    index += 1
                else:
                    index = 0

        return insar, 0





class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.train_test_split = json.load(open(config['split_path'],'r'))
        self.valid_files = self.train_test_split[mode]['0']
        self.valid_files.extend(self.train_test_split[mode]['1'])
        self.records = []
        self.oversampling = config['oversampling']
        self.positives = []
        self.negatives = []
        self.frameIDs = []
        self.augmentations = augmentations.get_augmentations(config)
        
        for file in self.valid_files:
            annotation = json.load(open(file,'r'))
            sample = file.split('/')[-1]
            insar_path = self.config['supervised_data_path'] + '/labeled/'+str(annotation['Deformation'][0])+'/' + sample[:-5] + '.png'
            mask_path = self.config['supervised_data_path'] + '/masks/'+str(annotation['Deformation'][0])+'/' + sample[:-5] + '.png'

            if len(annotation['Deformation'])>1:
                for activity in annotation['Deformation']:
                    insar_path = self.config['supervised_data_path'] + '/labeled/'+str(activity)+'/' + sample[:-5] + '.png'
                    mask_path = self.config['supervised_data_path'] + '/masks/'+str(activity)+'/' + sample[:-5] + '.png'

                    if os.path.isfile(insar_path):
                        break
            record = {'frameID':annotation['frameID'],'label':annotation['Deformation'],'intensity':annotation['Intensity'],'phase':annotation['Phase'],'insar_path':insar_path, 'mask_path':mask_path}
            if 0 not in record['label']:
                self.positives.append(record)
            else:
                self.negatives.append(record)
            self.records.append(record)
            self.frameIDs.append(record['frameID'])
        self.num_examples = len(self.records)
        self.frameIDs = np.unique(self.frameIDs)
        self.frame_dict = {}
        for idx, frame in enumerate(self.frameIDs):
            self.frame_dict[frame] = idx
        
        print(mode + ' Number of ground deformation samples: ',len(self.positives))
        print(mode + ' Number of non deformation samples: ',len(self.negatives))
    def __len__(self):
        return self.num_examples

    def read_insar(self, path):
        insar = cv.imread(path, 0)
        if insar is None:
            print("None")
            return insar
        if self.config['augment'] and self.mode == 'train':
            transform = self.augmentations(image=insar)
            insar = transform['image']
        insar = einops.repeat(insar, 'h w -> c h w', c=3)
        insar = torch.from_numpy(insar)/255
        return insar
    
    def __getitem__(self, index):
        label = None
        if self.oversampling and self.mode=='train':
            choice = random.randint(0,1)
            if choice == 0:
                choice_neg = random.randint(0,len(self.negatives)-1)
                sample = self.negatives[choice_neg]
            else:
                
                choice_pos = random.randint(0,len(self.positives)-1)
                sample = self.positives[choice_pos]
        else:
            sample = self.records[index]
       
        insar = self.read_insar(sample['insar_path'])
       
            
        if label is None:
            if not os.path.isfile(sample['mask_path']) and 0 in sample['label']:
                mask = np.zeros((224,224))
            else:
                mask = cv.imread(sample['mask_path'],0)
            
            if self.config['num_classes']<=2:
                if np.sum(mask)>0:
                    label = 1.0
                else:
                    label = 0.0
                if self.config['num_classes']==2:
                    
                    label = torch.tensor(label).long()
                    
                    return (insar.float(), torch.from_numpy(mask).long(), label)
                else:
                    label = torch.tensor(label).float()
                    return (insar.float(), torch.from_numpy(mask).float(), label)
            else:
                one_hot = torch.zeros((13))
                for act in sample['label']:
                    one_hot[act] = 1
                
                one_hot[sample['intensity']+7] = 1
                one_hot[sample['phase'] + 10] = 1
                
                return insar,mask, one_hot    
        else:
            label = torch.tensor(label).long()
            mask = np.zeros((224,224))
            return (insar.float(), torch.from_numpy(mask).long(), label)
