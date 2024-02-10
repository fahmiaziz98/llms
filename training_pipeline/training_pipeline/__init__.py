import os
import logging
import logging.config
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)



def initialize_logger(config_path: str="logging.yaml", logs_dir_name: str="logs") -> logging.Logger:
    """ Initialize the logger using a configuration file"""

    config_path_parent = Path(config_path).parent
    logs_dir = config_path_parent / logs_dir_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "rt") as f:
        config = yaml.safe_load(f.read())

    config["disable_existing_loggers"] = False
    logging.config.dictConfig(config)

def initialize(logging_config_path: str="logging.yaml", env_file_path: str=".env"):
    """
    Initialize the logger with a given configuration file and an optional environment variable file"
    
    Args:
        logging_config_path (str): The path to the logging configuration file. Defaults to "logging.yaml".
        env_file_path (str): The path to the environment variables file. Defaults to ".env".
    """
    
    logger.info("Initializing logger...")
    try:
        initialize_logger(config_path=logging_config_path)
    except FileNotFoundError:
        logger.info(
            f"Could not locate logging configuration file at {logging_config_path}"
        )

    logger.info("Initializing environment variables...")
    if env_file_path is None:
        env_file_path = find_dotenv(raise_error_if_not_found=True, usecwd=False)

    logger.info(f"Loading environment variables from: {env_file_path}")
    found_env_file = load_dotenv(env_file_path, verbose=True, override=True)
    if found_env_file is False:
        raise RuntimeError(f"Could not find environment file at: {env_file_path}")

    # Enable logging of model checkpoints.
    os.environ["COMET_LOG_ASSETS"] = "True"
    # Set to OFFLINE to run an Offline Experiment or DISABLE to turn off logging
    os.environ["COMET_MODE"] = "ONLINE"
    # Find out more about Comet ML configuration here: https://www.comet.com/docs/v2/integrations/ml-frameworks/huggingface/#configure-comet-for-hugging-face
