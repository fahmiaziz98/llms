"""
Preferensi:
    https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from comet_ml import API
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from training_pipeline import constants

logger = logging.getLogger(__name__)


def qlora_model(
        model_path: str="mistralai/Mistral-7B-v0.1",
        peft_model: Optional[str]=None,
        gradient_checkpoint: bool=True,
        cache_dir: Optional[Path]=None
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, PeftConfig]:
    """
    Function that builds a QLoRA LLM model.
    Args:
        model_path: Use the pretrained MISTRAL-7B v0.1 from Hugging Face Model Hub.
        peft_model: The name or path of the pretrained model to use for PeftModel.
        gradient_checkpointing (bool): Whether to use gradient checkpointing or not.
        cache_dir (Optional[Path]): The directory to cache the model in.
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]: A tuple containing the built model, tokenizer, and PeftConfig.
    """
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision="main",
        quantization_config=bnb_config,
        load_in_4bit=True,
        device_map="auto",
        trust_remote=False,
        cache_dir=str(cache_dir) if cache_dir else None
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
        truncation=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "</s>"})
        tokenizer.add_eos_token = True
        with torch.no_grad():
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token = tokenizer.pad_token
    
    if peft_model:
        is_model_name = not os.path.isdir(peft_model)
        if is_model_name:
            logger.info(
                f"Downloading {peft_model} from comet ml registry"
            )
            peft_model = download_from_registry(
                model_id=peft_model,
                cache_dir=cache_dir
            )
        logger.info(f"Load Lora config from {peft_model}")
        lora_config = LoraConfig.from_pretrained(peft_model)
        assert (
            lora_config.base_model_name_or_path == model_path
        ), f"Lora Model trained on different base model than the one requested: \
        {lora_config.base_model_name_or_path} != {model_path}"

        logger.info(f"Loading Peft Model from: {peft_model}")
        model = PeftModel.from_pretrained(model, peft_model)

    else:
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )

    if gradient_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = (
            False  # Gradient checkpointing is not compatible with caching.
        )
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True  # It is good practice to enable caching when using the model for inference.

    return model, tokenizer, lora_config


def download_from_registry(model_id: str, cache_dir:Optional[Path]=None):
    """
    Downloads a model from the Comet ML Learning model registry.

    Args:
        model_id (str): The ID of the model to download, in the format "workspace/model_name:version".
        cache_dir (Optional[Path]): The directory to cache the downloaded model in. Defaults to the value of
            `constants.CACHE_DIR`.

    Returns:
        Path: The path to the downloaded model directory.
    """
    if cache_dir is None:
        cache_dir = constants.CACHE_DIR
    output_folder = cache_dir / "models" / model_id

    already_downloaded = output_folder.exists()
    if not already_downloaded:
        workspace, model_id = model_id.split("/")
        model_name, version = model_id.split(":")

        api = API()
        model = api.get_model(workspace=workspace, model_name=model_name)
        model.download(version=version, output_folder=output_folder, expand=True)
    else:
        logger.info(f"Model {model_id=} already downloaded to: {output_folder}")

    subdirs = [d for d in output_folder.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        model_dir = subdirs[0]
    else:
        raise RuntimeError(
            f"There should be only one directory inside the model folder. \
                Check the downloaded model at: {output_folder}"
        )

    logger.info(f"Model {model_id=} downloaded from the registry to: {model_dir}")

    return model_dir


def prompt(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    device: str = "cuda:0",
    return_only_answer: bool = False,
):
    """
    Generates text based on the input text using the provided model and tokenizer.

    Args:
        model (transformers.PreTrainedModel): The model to use for text generation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for text generation.
        input_text (str): The input text to generate text from.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 40.
        temperature (float, optional): The temperature to use for text generation. Defaults to 1.0.
        device (str, optional): The device to use for text generation. Defaults to "cuda:0".
        return_only_answer (bool, optional): Whether to return only the generated text or the entire generated sequence.
            Defaults to False.

    Returns:
        str: The generated text.
    """

    inputs = tokenizer(input_text, return_tensors="pt", return_token_type_ids=False).to(
        device
    )

    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, temperature=temperature
    )

    output = outputs[
        0
    ]  # The input to the model is a batch of size 1, so the output is also a batch of size 1.
    if return_only_answer:
        input_ids = inputs.input_ids
        input_length = input_ids.shape[-1]
        output = output[input_length:]

    output = tokenizer.decode(output, skip_special_tokens=True)

    return output