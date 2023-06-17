import timm


def create_model(configs):
    if 'num_classes' in configs:
        num_classes = configs['num_classes']
    else:
        num_classes = 0
    if "vit" not in configs["architecture"]:
        model = timm.models.create_model(
            configs["architecture"].lower(), pretrained=True, num_classes=num_classes, in_chans=configs['num_channel']
        )
    else:
        model = timm.models.vision_transformer.VisionTransformer(
            img_size=int(configs["resolution"]),
            patch_size=int(configs["patches"]),
            in_chans=configs["num_channel"],
            embed_dim=configs["embed_dim"],
            depth=configs["depth"],
            num_heads=configs["num_heads"],
            num_classes=num_classes,
        )
    return model
