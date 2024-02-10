# import fire
from pathlib import Path



def train_sft(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
   
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
    from training_pipeline import initialize

    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)



if  __name__ == "__main__":
    train_sft()
