# Hephaestus Dataset

<p align="center">
  <img src="docs/hephaestus-logo.png">
</p>

This repository contains the data and code used in [Hephaestus: A large scale multitask dataset towards InSAR understanding](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Bountos_Hephaestus_A_Large_Scale_Multitask_Dataset_Towards_InSAR_Understanding_CVPRW_2022_paper.pdf) as published in CVPR 2022 workshop Earthvision.


If you use this work, please cite:
```
@InProceedings{Bountos_2022_CVPR,
    author    = {Bountos, Nikolaos Ioannis and Papoutsis, Ioannis and Michail, Dimitrios and Karavias, Andreas and Elias, Panagiotis and Parcharidis, Isaak},
    title     = {Hephaestus: A Large Scale Multitask Dataset Towards InSAR Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1453-1462}
}
```

### Contents
- [Setup](#dependancies)
- [Data and pretrained models](#dataset-and-pretrained-models)
- [SSL Training from scratch](#train-ssl-from-scratch)
- [Description of Hephaestus annotation scheme](#annotation)
- [Recreating cropped dataset](#reproduce-cropped-patches)
- [Acknowledgment](#acknowledgments)
### Dependancies
This repo has been tested with python3.9. To install the necessary dependancies run:   
`pip install -r requirements.txt`

### Dataset and pretrained models

The annotation files can be downloaded [here](https://www.dropbox.com/s/i08mz5514gczksz/annotations_hephaestus.zip?dl=0).

The raw InSAR data can be found [here](https://www.dropbox.com/s/uhyr4gefregqvdx/Hephaestus_Raw.zip?dl=0).

The dataset is organized in the following structure:
```
|- FrameID (e.g 124D_05291_081406)
|    |-- interferograms
|    |    |---  Dates (e.g 20181128_20181204)
|    |    |    |--- InSAR.png
|    |    |    |--- Coherence.png
```

The cropped 224x224 patches, along with the respective masks and labels can be found [here](https://www.dropbox.com/s/2bkpj79jepk0vks/Hephaestus_Classification.zip?dl=0).
Some examples of these patches can be seen in the following figure. 

![figure](docs/examples.png)

The directory structure for the cropped patches is:

```
|-- Root directory
|   |-- Class labels per patch id
|   |-- Masks
|   |-- Cropped patches
```
#### Pretrained encoders
The SSL (MoCov2) pretrained ResNet18 can be found [here](https://www.dropbox.com/s/t5w8x44p32gacf3/ResNet18-MocoV2.pth.tar?dl=0).

ResNet18 loading example:
```python
model = torch.load('ResNet18_Encoder.pt')
```

#### Train SSL from scratch
We currently support only MoCov2 (functionality for more will be added in the future). To choose a backbone modify the "architecture" field in the configs/configs.json.
Add your wandb-project and entity and the path of the dataset. If you want to continue from a checkpoint add the checkpoint's path at the "resume_checkpoint" field, otherwise insert false.

To modify the augmentation set as well as the settings for MoCoV2, edit the respective configuration files in config/augmentations and config/method/mocov2.

To initiate the training procedure run:
```python
python main.py
```
The script will automatically create folders for the checkpoints and store the config file and the wandb run id.


### Multi-GPU / Multi-Node training
You can make use of torchrun or SLURM to launch distributed jobs.

#### torchrun: 
Single-Node Multi-GPU:
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py
```

Multi-Node Multi-GPU:
```
# On XXX.XXX.XXX.62 (the master node)
torchrun \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
--master_addr=XXX.XXX.XXX.62 --master_port=1234 \
main.py

# On XXX.XXX.XXX.63 (the worker node)
torchrun \
--nproc_per_node=2 --nnodes=2 --node_rank=1 \
--master_addr=XXX.XXX.XXX.62 --master_port=1234 \
main.py
```

#### SLURM:
After setting the relevant parameters inside hephaestus.slurm:
```
sbatch hephaestus.slurm
```

### Annotation

The dataset contains both labeled and unlabeled data. The labeled part covers 38 frames summing up to 19,919 annotated InSAR.
The list of the studied volcanoes, along with the temporal distribution of their samples can be seen below. ![below](docs/volcano_distribution.png)

Each labeled InSAR is accompanied by a json file containing the annotation details. Below we present an example of an annotation file. A detailed description can be seen in the original paper (section 2.2):
```python
{
  "uniqueID": 19912, #ID given to the annotation file.
  "frameID": "103A_07010_121313", #ID of the InSARs frame. This is a unique location identifier.
  "primary_date": "20210414", #Caption date of the primary SAR image.
  "secondary_date": "20210426", #Caption date of the secondary SAR image.
  "corrupted": 0, #Flag for corrupted InSAR.
  "processing_error": 0, #Error attributed to InSAR processing.
  "glacier_fringes": 0, #Flag to identify the existence of glaciers
  "orbital_fringes": 0, #Fringes attributed to orbital errors.
  "atmospheric_fringes": 2, #Fringes attributed to atmospheric effects. Its value ranges from 0 to 3 with 0 denoting is absense.
  "low_coherence": 0, #Flag to denote low coherence.
  "no_info": 0, #Flag to identify InSAR with very low information signal.
  "image_artifacts": 0, #Flag to identify InSAR with artificats unrelated to interferograms.
  "label": [ 
    "Non_Deformation" #Labels. May contain multiple elements.
  ],
  "activity_type": [], #Activity type of each ground deformation pattern.
  "intensity_level": ["None"], #Intensity level of each activity type.
  "phase": "Rest", #Phase of the volcano. Rest/Unrest/Rebound.
  "confidence": 0.8, #Confidence of the annotator for this annotation.
  "segmentation_mask": [], #List of polygons containing the ground deformation patterns.
  "is_crowd": 0, #Flag to denote whether there are multiple ground deformation patterns in the InSAR.
  "caption": "Turbulent mixing effect or wave-like patterns caused by liquid and solid particles of the atmosphere can be detected around the area. No deformation activity can be detected."
}
```

### Annotation Processing

All the necessary utilities are contained in annotation_utils. One can directy load the annotation file or the segmentation mask(s) of the desired interferogram.

#### Segmentation mask example:

```
from annotation_utils import *

root_path = 'Your Root Path/'
annotation = root_path + 'annotations/10478.json'
get_segmentation(annotation,verbose=True)

```

#### Reproduce cropped patches

If needed one can reproduce the cropped patches with the desired resolution using the save_crops utility. 
Example usage:
```
from annotation_utils import *

root_path = 'Your Root Path/'
mask_path = root_path + 'Hephaestus/masks/'
save_path = root_path + 'Hephaestus/labeled/'
raw_insar_path = root_path + 'Hephaestus_Raw/'
annotation_folder = root_path + 'annotations/'
save_crops(annotation_folder=annotation_folder,save_path=save_path,mask_path=mask_path,raw_insar_path=raw_insar_path,out_size=224,verbose=False)
```

The functions 
`crop_around_object(annotation_path,verbose=True,output_size=224,index=0)` and `image_tiling(image,tile_size=224)` handle the cropping of the interferograms. When the InSAR of interest contains ground deformation the function `crop_around_object` is called, otherwise the InSAR is split in non-overlapping patches with `image_tiling`. 

`crop_around_object` makes sure to include the deformation pattern in a random crop at the desired resolution without excluding the presence of multiple ground deformation types in the cropped patch. If the desired output size is smaller than the area of a deformation mask, a warning is issued and the mask is cropped accordingly.




#### Acknowledgments
Special thanks to Alexandros Makridis for the Hephaestus logo.

This work is part of the project DeepCube and has been funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101004188
