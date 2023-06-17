import datetime
import json
import os
import shutil
from pathlib import Path

import timm
import torch
import torch.distributed as dist
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall

import dataset.Dataset as Dataset
import self_supervised.mocov2.builder as builder


def create_checkpoint_directory(args,wandb_run=None):
    if wandb_run is None:
        if 'num_classes' not in args:
            checkpoint_path = (
                Path("checkpoints")
                / args["method"].lower()
                / args["architecture"].lower()
                / f'{args["architecture"].lower()}_{str(args["resolution"])}'
                / str(datetime.datetime.now())
            )
        else:
            checkpoint_path = (
                Path("checkpoints")
                / "supervised"
                / args["architecture"].lower()
                / str(datetime.datetime.now())
            )
    else:
        if 'num_classes' not in args:
            checkpoint_path = (
                Path("checkpoints") / wandb_run
                / args["method"].lower()
                / args["architecture"].lower()
                / f'{args["architecture"].lower()}_{str(args["resolution"])}'
                /str(datetime.datetime.now())
            )
        else:
            checkpoint_path = (
                Path("checkpoints") / wandb_run
                / "supervised"
                / args["architecture"].lower()
                /str(datetime.datetime.now())
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
    return [accuracy, fscore, precision, recall]


def is_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ("LOCAL_RANK", "SLURM_LOCALID"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "SLURM_PROCID"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "SLURM_NTASKS"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def is_global_master(args):
    return args["rank"] == 0

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

def load_checkpoint(model, optimizer, args):
    if os.path.isfile(args["resume_checkpoint"]):
        print("=> loading checkpoint '{}'".format(args["resume_checkpoint"]))
        checkpoint = torch.load(args["resume_checkpoint"], map_location="cpu")
        args["start_epoch"] = checkpoint["epoch"]

        for key in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][key.replace("module.", "")] = checkpoint[
                "state_dict"
            ][key]
            del checkpoint["state_dict"][key]

        msg = model.load_state_dict(checkpoint["state_dict"])
        print(msg)

        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args["resume_checkpoint"], checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args["resume_checkpoint"]))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args["lr"]
    if args["cos"]:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args["epochs"]))
    else:  # stepwise lr schedule
        for milestone in args["schedule"]:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # print(correct[:k].shape)
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = (
                correct[:k].reshape(k * correct.shape[1]).float().sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))
        return res