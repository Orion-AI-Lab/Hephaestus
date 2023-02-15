import json
from pathlib import Path


def create_checkpoint_directory(args):
    checkpoint_path = (
        Path("checkpoints")
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
