import json
import pprint
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import wandb
from torchmetrics.classification import BinaryStatScores
from tqdm import tqdm

from utilities import utils as utils

np.random.seed(999)
torch.manual_seed(999)
random.seed(999)


def define_checkpoint(configs):
    checkpoint_path = (
        Path("finetuning_checkpoints")
        / configs["task"].lower()
        / configs["architecture"].lower()
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def eval_cls(configs, loader, criterion, mode="Val", model=None, epoch=-1):
    print("\n\nInitializing Evaluation: " + mode + "\n\n\n")
    if model is None:
        checkpoint_path = "best_model_task_" + configs["task"] + ".pt"
        model = torch.load(configs["checkpoint_path"] / checkpoint_path)["model"]
    model.eval()
    (
        accuracy,
        fscore,
        precision,
        recall,
        _,
        average_precision,
    ) = utils.initialize_metrics(configs)
    metric = BinaryStatScores().to(configs["device"])
    loss = 0.0
    prediction_list = []
    ground_truth = []

    with torch.no_grad():
        for batch in tqdm(loader):
            insar, _, label = batch
            insar = insar.to(configs["device"])

            label = label.to(configs["device"])
            out = model(insar)
            if configs["num_classes"] == 1:
                label = label.unsqueeze(1)
            if configs["multilabel"] or configs["num_classes"] == 1:
                predictions = torch.sigmoid(out)
                tmp_pred = predictions.clone()

                predictions[tmp_pred >= configs["threshold"]] = 1
                predictions[tmp_pred < configs["threshold"]] = 0
            else:
                predictions = out.argmax(1)

            prediction_list.extend(predictions.cpu().detach().numpy())
            ground_truth.extend(label.cpu().detach().numpy())

            accuracy(predictions, label)
            fscore(predictions, label)
            precision(predictions, label)
            recall(predictions, label)
            average_precision(out[:, 1].float(), label)

            metric(predictions, label)
            loss += criterion(out, label)
    print("TP, FP, TN, FN, Support")
    print(metric.compute())

    if configs["wandb"]:
        if configs["linear_evaluation"]:
            mode = mode + " LE"
        else:
            mode = mode + " FT"
        wandb.log(
            {
                mode + " F-Score": fscore.compute(),
                mode + " Acc": accuracy.compute(),
                mode + " Precision": precision.compute(),
                mode + " Recall": recall.compute(),
                mode + " Avg. Precision": average_precision.compute(),
                mode + "Loss: ": loss / len(loader.dataset),
            }
        )
    print(
        {
            mode + " F-Score": fscore.compute(),
            mode + " Acc": accuracy.compute(),
            mode + " Precision": precision.compute(),
            mode + " Recall": recall.compute(),
            mode + " Avg. Precision": average_precision.compute(),
        },
        mode + "Loss: ",
        loss / len(loader.dataset),
        "Epoch: ",
        epoch,
    )

    return fscore.compute()


def train_cls(configs):
    train_loader, val_loader, test_loader = utils.prepare_supervised_learning_loaders(
        configs
    )

    accuracy, fscore, precision, recall, _, avg = utils.initialize_metrics(configs)
    class_weights = torch.tensor(configs["class_weights"]).to(configs["device"])
    if configs["num_classes"] >= 2 or configs["num_classes"] == 1:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()

    if configs["ssl_encoder"] is None:
        print("=" * 20)
        print("Creating model pretrained on Imagenet")
        print("=" * 20)
        model = timm.create_model(
            configs["architecture"].lower(),
            num_classes=configs["num_classes"],
            pretrained=True,
        )
        if configs["linear_evaluation"]:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        if "vit" in configs["architecture"] or "swin" in configs["architecture"]:
            model.head = nn.Linear(model.head.in_features, configs["num_classes"])
        else:
            model.fc = nn.Linear(model.fc.in_features, configs["num_classes"])
    else:
        print("=" * 20)
        print("Creating SSL model pretrained on Hephaestus")
        print("=" * 20)
        id_json = json.load(open(configs["ssl_run_id_path"], "r"))
        configs["wandb_id"] = id_json["wandb_id"]
        print("SSL Wandb ID: ", configs["wandb_id"])
        if configs["wandb"]:
            wandb.init(
                project=configs["wandb_project"],
                entity=configs["wandb_entity"],
                id=configs["wandb_id"],
                resume=True,
            )
        model = torch.load(configs["ssl_encoder"], map_location=configs["device"])
        print(model)
        out_dim = 2048

        if configs["linear_evaluation"]:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
        else:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Linear(out_dim, configs["num_classes"])
    model.to(configs["device"])
    # print(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"]
    )

    best_val = 0.0
    best_epoch = 0.0
    best_stats = {}
    for epoch in range(configs["epochs"]):
        for idx, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            if not configs["linear_evaluation"]:
                model.train()
            insar, _, label = batch
            insar = insar.to(configs["device"])
            label = label.to(configs["device"])
            if configs["num_classes"] == 1:
                label = label.unsqueeze(1)
            out = model(insar)
            if configs["multilabel"] or configs["num_classes"] == 1:
                predictions = torch.sigmoid(out)
                tmp_pred = predictions.clone()
                predictions[tmp_pred >= configs["threshold"]] = 1
                predictions[tmp_pred < configs["threshold"]] = 0
            else:
                predictions = out.argmax(1)
            accuracy(predictions, label)

            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                if configs["linear_evaluation"]:
                    log_dict = {
                        "LE Loss: ": loss.mean().item(),
                        "LE Train accuracy: ": accuracy.compute(),
                        "LE Epoch": epoch,
                    }
                else:
                    log_dict = {
                        "FT Loss: ": loss.mean().item(),
                        "FT Train accuracy: ": accuracy.compute(),
                        "FT Epoch": epoch,
                    }
                print(log_dict)
                if configs["wandb"]:
                    wandb.log(log_dict)

        val_loss = eval_cls(configs, val_loader, criterion, model=model, epoch=epoch)
        if val_loss >= best_val:
            best_val = val_loss
            best_epoch = epoch
            best_stats = {"val_loss": best_val, "epoch": best_epoch}
            print("New Best model: ", best_stats)

            model_path = "best_model_task_" + configs["task"] + ".pt"
            print("Saving model to: ", configs["checkpoint_path"] / model_path)
            torch.save(
                {"model": model, "stats": best_stats},
                configs["checkpoint_path"] / model_path,
            )

    print("=" * 20)
    print("Start Testing")
    print("=" * 20)
    eval_cls(configs, test_loader, criterion, mode="Test")


def train_segmentation(configs):
    pass


if __name__ == "__main__":
    config_path = "configs/supervised_configs.json"
    configs = json.load(open(config_path, "r"))

    augmentation_cfg = json.load(open("configs/augmentations/augmentation.json", "r"))
    configs.update(augmentation_cfg)
    configs["checkpoint_path"] = define_checkpoint(configs)
    if configs["wandb"] and not configs["ssl_encoder"]:
        wandb.init(
            project=configs["wandb_project"],
            entity=configs["wandb_entity"],
            config=configs,
        )
    pprint.pprint(configs)
    train_cls(configs)
