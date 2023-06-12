import json
from pathlib import Path

import self_supervised.mocov2.builder as builder
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import Dataset
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall

def create_checkpoint_directory(args,wandb_run=None):
    if wandb_run is None:
        checkpoint_path = (
            Path("checkpoints")
            / args["method"].lower()
            / args["architecture"].lower()
            / f'{args["architecture"].lower()}_{str(args["resolution"])}'
        )
    else:
        checkpoint_path = (
            Path("checkpoints") / wandb_run
            / args["method"].lower()
            / args["architecture"].lower()
            / f'{args["architecture"].lower()}_{str(args["resolution"])}'
        )
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def prepare_configuration(path):
    # Load configuration files
    base_cfg = json.load(open(path, "r"))
    if not base_cfg["seed"]:
        base_cfg["seed"] = None

    augmentation_cfg = json.load(open(base_cfg["augmentation_config"], "r"))
    base_cfg.update(augmentation_cfg)

    model_cfg = (
        Path("configs/method")
        / base_cfg["method"].lower()
        / f'{base_cfg["method"].lower()}.json'
    )
    with model_cfg.open("r", encoding="UTF-8") as target:
        model_config = json.load(target)
    base_cfg.update(model_config)

    # Create checkpoint path if it does not exist
    checkpoint_path = create_checkpoint_directory(base_cfg)
    base_cfg["checkpoint_path"] = checkpoint_path.as_posix()

    return base_cfg

def load_encoder(checkpoint_path,config=None):
    if config is None:
        config_path = "configs/configs.json"
        config = prepare_configuration(config_path)
        config['pretrained']=False
    checkpoint = torch.load(checkpoint_path,map_location='cpu')
    if config["method"] == "mocov2":
        model = builder.MoCo(
            config,
            config["moco_dim"],
            config["moco_k"],
            config["moco_m"],
            config["moco_t"],
            config["mlp"])
        print(model)
        dist.init_process_group(backend="nccl")

        model = nn.parallel.DistributedDataParallel(model.cuda())
        model.load_state_dict(checkpoint['state_dict'])
        model.module.encoder_q.fc = nn.Identity()
        torch.save(model.module.encoder_q,'pretrained_encoder_'+config['architecture']+'.pt')
        return model.module.encoder_q



def prepare_supervised_learning_loaders(configs):
    train_dataset = Dataset.FullFrameDataset(config=configs,mode='train')
    val_dataset = Dataset.FullFrameDataset(config=configs,mode='val')
    test_dataset = Dataset.FullFrameDataset(config=configs,mode='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=configs['num_workers'],pin_memory=True,
                                     drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=configs['num_workers'],pin_memory=True,
                                     drop_last=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=configs['num_workers'], pin_memory=True,
                                     drop_last=False)
    
    return train_loader, val_loader, test_loader

def initialize_metrics(configs):
    accuracy = Accuracy(task='multiclass', average='micro',multidim_average='global',num_classes=configs['num_classes']).to(configs['device'])
    fscore = F1Score(task='multiclass', num_classes=configs['num_classes'],average='micro',multidim_average='global').to(configs['device'])
    precision = Precision(task='multiclass', average='micro', num_classes=configs['num_classes'],multidim_average='global').to(configs['device'])
    recall = Recall(task='multiclass', average='micro', num_classes=configs['num_classes'],multidim_average='global').to(configs['device'])
    return accuracy, fscore, precision, recall