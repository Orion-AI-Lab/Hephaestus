import json
from pathlib import Path

import self_supervised.mocov2.builder as builder
import timm
import torch
import torch.distributed as dist
import torch.nn as nn


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