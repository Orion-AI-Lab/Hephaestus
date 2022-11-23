import torch
import torchvision
import albumentations as A

augmentation_dict = {
    'RandomResizedCrop':{'func':A.augmentations.crops.transforms.RandomResizedCrop,'values':['height','width']},
    'ColorJitter':{'func':A.augmentations.transforms.ColorJitter,'values':['brighness','contrast','saturation','hue']},
    'HorizontalFlip':{'func':A.augmentations.transforms.HorizontalFlip,'values':[]},
    'GaussianBlur':{'func':A.augmentations.transforms.GaussianBlur,'values':['sigma_limit']},
    "ElasticTransform":{'func':A.augmentations.geometric.transforms.ElasticTransform,'values':""}
}


def get_augmentations(config):
    augmentations = config['augmentations']
    independend_aug = []
    for k,v in augmentations.items():
        if k == 'RandomResizedCrop':
            aug = A.augmentations.crops.transforms.RandomResizedCrop(height=v['value'],width=v['value'],p=v['p'])
        elif k=='ColorJitter':
            aug = A.augmentations.transforms.ColorJitter(brightness=v['value'][0],contrast=v['value'][1],saturation=v['value'][2],hue=v['value'][3],p=v['p'])
        elif k=='HorizontalFlip':
            aug = A.augmentations.transforms.HorizontalFlip(p=v['p'])
        elif k=='VerticalFlip':
            aug = A.augmentations.transforms.VerticalFlip(p=v['p'])
        elif k=='GaussianBlur':
            aug = A.augmentations.transforms.GaussianBlur(sigma_limit=v['value'],p=v['p'])
        elif k=='ElasticTransform':
            aug = A.augmentations.geometric.transforms.ElasticTransform(p=v['p'])
        elif k=='Cutout':
            aug = A.augmentations.dropout.coarse_dropout.CoarseDropout(p=v['p'])
        elif k=='GaussianNoise':
            aug = A.augmentations.transforms.GaussNoise(p=v['p'])
        elif k=='MultNoise':
            aug = A.augmentations.transforms.MultiplicativeNoise(p=v['p'])
        independend_aug.append(aug)
    return A.Compose(independend_aug)