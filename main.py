import json
from self_supervised.mocov2 import builder
from utilities.utils import prepare_configuration
from training import train_ssl, train_supervised
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervised', action='store', type=bool, default=True)
    args = parser.parse_args()
    if args.supervised:
        config_path = "configs/supervised_configs.json"
        configs = json.load(open(config_path,'r'))
        train_supervised.train(configs)
        train_supervised.test(configs,phase='test')
    else:
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

        train_ssl.exec_model(model, config)
