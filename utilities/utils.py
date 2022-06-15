from pathlib import Path
import json

def create_checkpoint_directory(args):
    checkpoint_path = 'checkpoints/' + args['method'].lower() + '/' + args['architecture'].lower() + '/' + args['architecture'].lower() + '_'+ str(args['resolution'])
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    return checkpoint_path

def announce_stuff(announcement,symbol='=',times=20,up=True,down=True):
    if up:
        print(symbol*times)
    print(announcement)
    if down:
        print(symbol*times)

def prepare_configuration(path):
    config = json.load(open(path, 'r'))
    # Create checkpoint path if it does not exist
    checkpoint_path = create_checkpoint_directory(config)
    config['checkpoint_path'] = checkpoint_path

    # Load augmentation settings
    augmentation_config = json.load(open(config['augmentation_config'],'r'))
    config.update(augmentation_config)

    #Load model settings
    model_config_path = 'configs/method/' + config['method'].lower() + '/' + config['method'].lower() + '.json'
    model_config = json.load(open(model_config_path,'r'))
    config.update(model_config)

    if config['seed'] == '':
        config['seed'] = None
    if config['gpu']==-1:
        config['gpu'] = None

    return config