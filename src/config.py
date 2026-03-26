import yaml

def get_config() -> dict:
    """
    Loads a YAML configuration file and returns it as a dictionary

    Returns:
    - dict: The configuration loaded from the YAML file
    """

    with open("./config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    return config

config = get_config()