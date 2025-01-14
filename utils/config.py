import yaml

from utils.print_log import print_config

def read_config(config_file_name: str):
    with open(config_file_name, 'r') as file:
        config = yaml.safe_load(file)
    
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    print_config(config)

    return config