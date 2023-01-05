import albumentations as A

def get_augmentations(config):
    augmentations = config['augmentations']
    independend_aug = []
    for k,v in augmentations.items():
        if k == 'RandomResizedCrop':
            aug = A.augmentations.RandomResizedCrop(height=v['value'],width=v['value'],p=v['p'])
        elif k=='ColorJitter':
            aug = A.augmentations.ColorJitter(brightness=v['value'][0],contrast=v['value'][1],saturation=v['value'][2],hue=v['value'][3],p=v['p'])
        elif k=='HorizontalFlip':
            aug = A.augmentations.HorizontalFlip(p=v['p'])
        elif k=='VerticalFlip':
            aug = A.augmentations.VerticalFlip(p=v['p'])
        elif k=='GaussianBlur':
            aug = A.augmentations.GaussianBlur(sigma_limit=v['value'],p=v['p'])
        elif k=='ElasticTransform':
            aug = A.augmentations.ElasticTransform(p=v['p'])
        elif k=='Cutout':
            aug = A.augmentations.CoarseDropout(p=v['p'])
        elif k=='GaussianNoise':
            aug = A.augmentations.GaussNoise(p=v['p'])
        elif k=='MultNoise':
            aug = A.augmentations.MultiplicativeNoise(p=v['p'])
        independend_aug.append(aug)
    return A.Compose(independend_aug)