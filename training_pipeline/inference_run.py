from pathlib import Path
import fire

from training_pipeline.training_pipeline import configs


def inference(
        config_file: str,
        dataset_dir: str,
        output_dir: str = "output-inference",
        env_file_path: str = ".env",
        logging_config_path: str = "logging.yaml",
        model_chache_dir: str = None
):
    import logging
    from training_pipeline.training_pipeline import initialize
    from training_pipeline.training_pipeline import utils
    from training_pipeline.training_pipeline.api import InferenceAPI

    logger = logging.getLogger(__name__)

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    config_file = Path(config_file)
    root_dataset_dir = Path(dataset_dir)
    model_cache_dir = Path(model_cache_dir) if model_cache_dir else None
    inference_output_dir = Path(output_dir)
    inference_output_dir.mkdir(exist_ok=True, parents=True)

    inference_config = configs.InferenceConfig.from_yaml(config_file)
    inference_api = InferenceAPI.from_config(
        config=inference_config,
        root_dataset_dir=root_dataset_dir,
        model_cache_dir=model_cache_dir,
    )
    inference_api.infer_all(
        output_file=inference_output_dir / "output-inference-api.json"
    )


if __name__ == "__main__":
    fire.Fire(inference)