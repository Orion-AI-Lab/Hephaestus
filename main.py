import builtins
import json
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import wandb

import Dataset
from self_supervised.mocov2 import builder
from utilities.utils import prepare_configuration


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


def train(train_loader, model, criterion, optimizer, epoch, args):
    print("Training epoch: ", epoch)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args["print_frequency"] == 0:
            progress.display(i)
            if is_global_master(args):
                wandb.log(
                    {
                        "top1": top1.avg,
                        "top5": top5.avg,
                        "loss": loss.item(),
                        "epoch": epoch,
                        "iteration": i,
                    }
                )


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


def exec_model(model, args):
    if args["seed"] is not None:
        random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if is_distributed():
        if "SLURM_PROCID" in os.environ:
            args["local_rank"], args["rank"], args["world_size"] = world_info_from_env()
            args["num_workers"] = int(os.environ["SLURM_CPUS_PER_TASK"])
            os.environ["LOCAL_RANK"] = str(args["local_rank"])
            os.environ["RANK"] = str(args["rank"])
            os.environ["WORLD_SIZE"] = str(args["world_size"])
            dist.init_process_group(
                backend="nccl",
                world_size=args["world_size"],
                rank=args["rank"],
            )
            os.environ["WANDB_MODE"] = "offline"
        else:
            args["local_rank"], _, _ = world_info_from_env()
            dist.init_process_group(backend="nccl")
            args["world_size"] = dist.get_world_size()
            args["rank"] = dist.get_rank()

        torch.cuda.set_device(args["local_rank"])

        # suppress printing if not master
        if not is_global_master(args):

            def print_pass(*args):
                pass

            builtins.print = print_pass
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    if is_global_master(args):
        # Initialize wandb
        print("Initializing Wandb")
        if args["resume_wandb"]:
            id_json = json.load(open(args["checkpoint_path"] + "/id.json"))
            args["wandb_id"] = id_json["wandb_id"]
            wandb.init(
                project=args["wandb_project"],
                entity=args["wandb_entity"],
                id=args["wandb_id"],
                resume=args["resume_wandb"],
            )
        else:
            id = wandb.sdk.lib.runid.generate_id()
            args["wandb_id"] = id
            wandb.init(
                project=args["wandb_project"],
                entity=args["wandb_entity"],
                config=args,
                id=id,
                resume="allow",
            )
            json.dump({"wandb_id": id}, open(args["checkpoint_path"] + "/id.json", "w"))
        wandb.watch(model)

    print("=> creating model '{}'".format(config["architecture"]))
    print(model)

    model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    if args["resume_checkpoint"]:
        load_checkpoint(model, optimizer, args)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args["local_rank"]]
    )

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    print("Initializing Dataset")
    train_dataset = Dataset.Dataset(args)
    print("Dataset initialized. Size: ", len(train_dataset))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=(train_sampler is None),
        num_workers=args["num_workers"],
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    for epoch in range(args["start_epoch"], args["epochs"]):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        if is_global_master(args):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args["architecture"],
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=args["checkpoint_path"]
                + "/checkpoint_{:04d}.pth.tar".format(epoch),
            )
    if is_global_master(args):
        wandb.finish()


if __name__ == "__main__":
    # Parse configurations
    config_path = "configs/configs.json"
    config = prepare_configuration(config_path)
    json.dump(config, open(config["checkpoint_path"] + "/config.json", "w"))

    if config["method"] == "mocov2":
        model = builder.MoCo(
            config,
            config["moco_dim"],
            config["moco_k"],
            config["moco_m"],
            config["moco_t"],
            config["mlp"],
        )
    else:
        raise NotImplementedError(f'{config["method"]} is not supported.')

    exec_model(model, config)
