import fire
from pathlib import Path

from training_pipeline.training_pipeline import configs



def train(
    config_file: str,
    output_dir: str,
    dataset_dir: str,
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cahce_dir: str = None
):
    """
    Fine Tuning model using Supervised Fine Tuning (SFT)
    Args:
        config_file (str): The path to the configuration file for the training process.
        output_dir (str): The directory where the trained model will be saved.
        dataset_dir (str): The directory where the training dataset is located.
        env_file_path (str, optional): The path to the environment variables file. Defaults to ".env".
        logging_config_path (str, optional): The path to the logging configuration file. Defaults to "logging.yaml".
        model_cache_dir (str, optional): The directory where the trained model will be cached. Defaults to None.
    """

    import logging
    from training_pipeline.training_pipeline import initialize
    from training_pipeline.training_pipeline import utils
    from training_pipeline.training_pipeline.api import TrainerAPI

    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    logger = logging.getLogger(__name__)
    logger.info("#"*100)
    utils.log_available_gpu()
    utils.log_available_ram()
    logger.info("#"*100)
    logger.info("Started Training Process")

    #  Load Configurations and Create Output Directory
    config_file = Path(config_file)
    output_dir = Path(output_dir)
    dataset_dir = Path(dataset_dir)
    model_cahce_dir = Path(model_cahce_dir) if model_cahce_dir else None

    training_config = configs.TrainingConfig.from_yaml(config_file, output_dir)
    trainer = TrainerAPI.from_config(
        config=training_config,
        root_dataset_dir=dataset_dir,
        model_cache_dir=model_cahce_dir
    )

    trainer.train()
    logger.info("Finished Training Process")
    
if __name__ == "__main__":
    fire.Fire(train)