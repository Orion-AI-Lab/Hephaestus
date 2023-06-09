import json
import os
import random

import cv2 as cv
import einops
import numpy as np
import torch
from tqdm import tqdm

from utilities import augmentations

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

        return insar


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.train_test_split = json.load(open(config["split_path"], "r"))
        self.valid_files = self.train_test_split[mode]["0"]
        self.valid_files.extend(self.train_test_split[mode]["1"])
        self.records = []
        self.oversampling = config["oversampling"]
        self.positives = []
        self.negatives = []
        self.frameIDs = []
        self.augmentations = augmentations.get_augmentations(config)
        root_path = self.config["cls_path"]
        for file in self.valid_files:
            annotation = json.load(open(root_path + file, "r"))
            sample = file.split("/")[-1]
            insar_path = (
                self.config["supervised_data_path"]
                + "/labeled/"
                + str(annotation["Deformation"][0])
                + "/"
                + sample[:-5]
                + ".png"
            )
            mask_path = (
                self.config["supervised_data_path"]
                + "/masks/"
                + str(annotation["Deformation"][0])
                + "/"
                + sample[:-5]
                + ".png"
            )

            if len(annotation["Deformation"]) > 1:
                for activity in annotation["Deformation"]:
                    insar_path = (
                        self.config["supervised_data_path"]
                        + "/labeled/"
                        + str(activity)
                        + "/"
                        + sample[:-5]
                        + ".png"
                    )
                    mask_path = (
                        self.config["supervised_data_path"]
                        + "/masks/"
                        + str(activity)
                        + "/"
                        + sample[:-5]
                        + ".png"
                    )

                    if os.path.isfile(insar_path):
                        break
            record = {
                "frameID": annotation["frameID"],
                "label": annotation["Deformation"],
                "intensity": annotation["Intensity"],
                "phase": annotation["Phase"],
                "insar_path": insar_path,
                "mask_path": mask_path,
            }
            if 0 not in record["label"]:
                self.positives.append(record)
            else:
                self.negatives.append(record)
            self.records.append(record)
            self.frameIDs.append(record["frameID"])
        self.num_examples = len(self.records)
        self.frameIDs = np.unique(self.frameIDs)
        self.frame_dict = {}
        for idx, frame in enumerate(self.frameIDs):
            self.frame_dict[frame] = idx

        print(mode + " Number of ground deformation samples: ", len(self.positives))
        print(mode + " Number of non deformation samples: ", len(self.negatives))

    def __len__(self):
        return self.num_examples

    def read_insar(self, path):
        insar = cv.imread(path, 0)
        if insar is None:
            print("None")
            return insar
        if self.config["augment"] and self.mode == "train":
            transform = self.augmentations(image=insar)
            insar = transform["image"]
        insar = einops.repeat(insar, "h w -> c h w", c=3)
        insar = torch.from_numpy(insar) / 255
        return insar

    def __getitem__(self, index):
        label = None
        if self.oversampling and self.mode == "train":
            choice = random.randint(0, 1)
            if choice == 0:
                choice_neg = random.randint(0, len(self.negatives) - 1)
                sample = self.negatives[choice_neg]
            else:
                choice_pos = random.randint(0, len(self.positives) - 1)
                sample = self.positives[choice_pos]
        else:
            sample = self.records[index]

        insar = self.read_insar(sample["insar_path"])

        if label is None:
            if not os.path.isfile(sample["mask_path"]) and 0 in sample["label"]:
                mask = np.zeros((224, 224))
            else:
                mask = cv.imread(sample["mask_path"], 0)

            if self.config["num_classes"] <= 2:
                if np.sum(mask) > 0:
                    label = 1.0
                else:
                    label = 0.0
                if self.config["num_classes"] == 2:
                    label = torch.tensor(label).long()

                    return (insar.float(), torch.from_numpy(mask).long(), label)
                else:
                    label = torch.tensor(label).float()
                    return (insar.float(), torch.from_numpy(mask).float(), label)
            else:
                one_hot = torch.zeros((13))
                for act in sample["label"]:
                    one_hot[act] = 1

                one_hot[sample["intensity"] + 7] = 1
                one_hot[sample["phase"] + 10] = 1

                return insar, mask, one_hot
        else:
            label = torch.tensor(label).long()
            mask = np.zeros((224, 224))
            return (insar.float(), torch.from_numpy(mask).long(), label)


class FullFrameDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode="train"):
        self.data_path = config["data_path"]
        self.config = config
        self.mode = mode
        if config["augment"]:
            self.augmentations = augmentations.get_augmentations(config)
        else:
            self.augmentations = None

        self.interferograms = []
        self.channels = config["num_channels"]
        annotation_path = "./annotations/"
        annotations = os.listdir(annotation_path)
        frames = os.listdir(self.data_path)
        unique_frames = np.unique(frames)
        test_frames = ["124D_04854_171313", "022D_04826_121209", "087D_07004_060904"]

        self.frame_dict = {}
        for idx, frame in enumerate(unique_frames):
            self.frame_dict[frame] = idx
        self.positives = []
        self.negatives = []
        for idx, annotation_file in tqdm(enumerate(annotations)):
            annotation = json.load(open(annotation_path + annotation_file, "r"))
            if annotation["frameID"] in test_frames and (
                mode == "train" or mode == "val"
            ):
                continue
            if annotation["frameID"] not in test_frames and mode == "test":
                continue
            if "Non_Deformation" in annotation["label"]:
                label = 0
            else:
                label = 1
            sample_dict = {
                "frameID": annotation["frameID"],
                "insar_path": annotation_utils.get_insar_path(
                    annotation_path=annotation_path + annotation_file,
                    root_path=self.config["data_path"],
                ),
                "label": annotation,
            }
            self.interferograms.append(sample_dict)
            if label == 0:
                self.negatives.append(sample_dict)
            else:
                self.positives.append(sample_dict)
        self.num_examples = len(self.interferograms)
        random.Random(999).shuffle(self.positives)
        random.Random(999).shuffle(self.negatives)
        if self.mode == "train":
            self.positives = self.positives[: int(0.9 * len * self.positives)]
            self.negatives = self.negatives[: int(0.9 * len(self.negatives))]
            self.interferograms = self.positives
            self.interferograms.extend(self.negatives)
        elif self.mode == "val":
            self.positives = self.positives[int(0.9 * len * self.positives) :]
            self.negatives = self.negatives[int(0.9 * len(self.negatives)) :]
            self.interferograms = self.positives
            self.interferograms.extend(self.negatives)
        print("Mode: ", self.mode, " Number of examples: ", self.num_examples)

    def __len__(self):
        return self.num_examples

    def prepare_insar(self, insar):
        insar = torch.from_numpy(insar).float().permute(2, 0, 1)
        insar /= 255
        normalize = torchvision.transforms.Normalize(
            mean=[0.5472, 0.7416], std=[0.4142, 0.2995]
        )
        insar = normalize(insar)
        return insar

    def read_insar(self, path):
        insar = cv.imread(path, 0)
        if insar is None:
            print("None")
            return insar, 0
        coherence_path = path[:-8] + "cc.png"
        coherence = cv.imread(coherence_path, 0)
        insar = einops.rearrange([insar, coherence], "c h w -> c h w")

        if insar is None:
            print("None")
            return insar, 0

        insar = self.prepare_insar(insar)
        return insar, 0

    def __getitem__(self, index):
        insar = None
        if self.config["oversampling"] and self.mode == "train":
            while insar is None:
                choice = random.randint(0, 1)
                if choice == 0:
                    choice_neg = random.randint(0, len(self.negatives) - 1)
                    sample = self.negatives[choice_neg]
                else:
                    choice_pos = random.randint(0, len(self.positives) - 1)
                    sample = self.positives[choice_pos]

                insar, _ = self.read_insar(sample["insar_path"])
        else:
            while insar is None:
                sample = self.interferograms[index]
                path = sample["insar_path"]

                insar, _ = self.read_insar(path)
                if insar is None:
                    if index < self.num_examples - 1:
                        index += 1
                    else:
                        index = 0
        annotation = sample["label"]
        if "Non_Deformation" in annotation["label"]:
            label = 0
        else:
            label = 1
        return insar, torch.tensor(label).long()
