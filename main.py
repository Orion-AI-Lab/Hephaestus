import torchvision
import sys
import numpy as np
import torch
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision
import time
import timm
import wandb
import kornia
import random
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import pprint
from tqdm import tqdm
from utilities.utils import *
from models.model_utils import *
import self_supervised
import self_supervised.mocov2
from self_supervised.mocov2 import mocov2 as moco
#os.environ['CUDA_VISIBLE_DEVICES'] ='1,2'
def exec_ssl(config):
    announce_stuff('Initializing ' + config['method'],up=False)
    if config['method']=='mocov2':
        moco.main(config)

if __name__ == '__main__':

    #Parse configurations
    config_path = 'configs/configs.json'
    config = prepare_configuration(config_path)
    json.dump(config,open(config['checkpoint_path']+'/config.json','w'))

    # Set seeds
    seed = config['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    #Initialize wandb

    if config['resume_wandb']:
        id_json = json.load(open(config['checkpoint_path']+'/id.json'))
        config['wandb_id']=id_json['wandb_id']
        wandb.init(project=config['wandb_project'],entity=config['wandb_entity'],id=config['wandb_id'],resume=config['resume_wandb'])
    else:
        announce_stuff('Initializing Wandb')
        id = wandb.util.generate_id()
        config['wandb_id'] = id
        wandb.init(project=config['wandb_project'], entity=config['wandb_entity'],config=config,id=id,resume='allow')
        json.dump({'wandb_id':id},open(config['checkpoint_path']+'/id.json','w'))

    announce_stuff('Starting project with the following settings:')
    pprint.pprint(config)
    announce_stuff('',up=False)
    exec_ssl(config)



