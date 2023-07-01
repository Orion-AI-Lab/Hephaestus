import builtins
import copy
import json
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms.functional as F
import wandb
import webdataset as wds
import mae_utils as misc
import mae_scheduler as lr_sched
from torchvision.transforms import (
    Compose,
    Grayscale,
    Normalize,
    RandomCrop,
    Resize,
    ToTensor,
)
import sys
import dataset.Dataset as Dataset
from self_supervised.mocov2 import builder
from utilities.utils import prepare_configuration, is_distributed, is_global_master, world_info_from_env, save_checkpoint, load_checkpoint, AverageMeter, ProgressMeter, adjust_learning_rate, accuracy


def train(train_loader, model, criterion, optimizer, epoch, args,loss_scaler):
    print("Training epoch: ", epoch)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # switch to train mode
    model.train()
    accum_iter = args['accum_iter']
    end = time.time()
    for i, (img1, _) in enumerate(train_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if i % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)
        img1 = img1.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(img1, mask_ratio=args['mask_ratio'])

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(i + 1) % accum_iter == 0)
        if (i + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if (i + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((i / len(train_loader) + epoch) * 1000)
            #log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            #log_writer.add_scalar('lr', lr, epoch_1000x)
            wandb.log({'train_loss':loss_value_reduce,'lr':lr,'epoch':epoch_1000x})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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

    print("=> creating model '{}'".format(args["architecture"]))
    print(model)

    model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args["lr"],
        momentum=args["momentum"],
        weight_decay=args["weight_decay"],
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
    if args["webdataset"]:
        from utilities.augmentations import get_augmentations

        def stack_and_augment(src):
            for sample in src:
                cc, diff = sample
                img1 = torch.cat((cc, diff), 0).permute(1, 2, 0).numpy()
                img2 = copy.deepcopy(img1)
                tranform = Compose(
                    [
                        ToTensor(),
                        Normalize(
                            mean=[0.5472, 0.7416],
                            std=[0.4142, 0.2995],
                        ),
                    ]
                )
                yield (
                    tranform(augmentations(image=img1)["image"]),
                    tranform(augmentations(image=img2)["image"]),
                )

        base_transform = Compose(
            [
                Resize(size=args["resolution"]),
                RandomCrop(size=args["resolution"]),
                Grayscale(),
            ]
        )

        augmentations = get_augmentations(args)

        global_batch_size = args["batch_size"] * args["world_size"]
        num_batches = math.floor(args["wds_size"] / global_batch_size)
        num_worker_batches = math.floor(num_batches / args["num_workers"])
        args["num_batches"] = num_worker_batches * args["num_workers"]

        train_dataset = (
            wds.DataPipeline(
                wds.ResampledShards(args["data_path"]),
                wds.tarfile_to_samples(),
                wds.shuffle(1000),
                wds.decode("torch"),
                wds.to_tuple("cc.png", "diff.png"),
                wds.map_tuple(base_transform, base_transform),
                stack_and_augment,
                wds.batched(args["batch_size"], partial=False),
            )
            .with_epoch(num_worker_batches)
            .with_length(args["wds_size"])
        )

        train_loader = wds.WebLoader(
            train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args["num_workers"],
            persistent_workers=True,
            pin_memory=True,
        )

    else:
        train_dataset = Dataset.Dataset(args)
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
        args["num_batches"] = len(train_loader)

    print("Dataset initialized. Size: ", len(train_dataset))

    for epoch in range(args["start_epoch"], args["epochs"]):
        if not args["webdataset"]:
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
