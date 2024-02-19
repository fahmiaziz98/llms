from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict, dataclass, field

from datasets import Dataset
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs

from training_pipeline.constants import Scope
from training_pipeline.data.utils import load_json
from training_pipeline.prompt_templates.prompt import get_template


@dataclass(frozen=True)
class DataSample:
    """
    A data sample for a question answering model.

    Attributes:
        instruction (str): The instruction for question
        chat_history (str): The chat history for the question.
        question (str): The question to be answered.
        answer (str): The answer to the question.
    """

    instruction: str=field(repr=False)
    chat_history: str=""  
    question: str=""
    answer: str=""

class FinanceDataset:
    """
    A class representing a finance dataset.

    Args:
        data_path (Path): The path to the data file.
        scope (Scope, optional): The scope of the dataset. Defaults to Scope.TRAINING.
        template (str, optional): The template to use for the dataset. Defaults to "falcon".
        max_samples (Optional[int], optional): The maximum number of samples to use. Defaults to None.
    """
    def __init__(self, data_path: Path, scope: Scope=Scope.TRAINING, template: str="mistralai", max_samples: Optional[int]=None):
        
        self._data_path = data_path
        self._templates = get_template(template)
        self._max_samples = max_samples
        self._scope = scope
        self._raw_data = self.load(data_path)

    def load(self, data_path: Path) -> List[DataSample]:
        """
        Loads the data from the specified path.

        Args:
            data_path (Path): The path to the data file.

        Returns:
            List[DataSample]: The loaded data.
        """
        data = load_json(data_path)
        if self._max_samples is not None:
            data = data[: self._max_samples]
        return self.deserialize(data)
    
    def deserialize(self, data: List[dict]) -> List[DataSample]:
        """
        Deserializes the data.

        Args:
            data (List[dict]): The data to deserialize.

        Returns:
            List[DataSample]: The deserialized data.
        """

        if self._scope == Scope.TRAINING:
            return [
                DataSample(
                    instruction=sample["instruction"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["input"],
                    answer=sample["output"]
                )
                for sample in data
            ]
        else:
            return [
                DataSample(
                    instruction=sample["instruction"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["input"]
                )
                for sample in data
            ]
        
    def clean(self, samples: Dict[str, str]) -> Dict[str, str]:
        """
        Clean the samples.

        Args:
            samples (Dict[str, str]): The samples to clean.

        Returns:
            Dict[str, str]: The cleaned samples.
        """

        for key, sample in samples.items():
            cleaned_sample = clean_extra_whitespace(sample)
            cleaned_sample = group_broken_paragraphs(cleaned_sample)

            samples[key] = cleaned_sample

        return samples
    
    def to_huggingface(self) -> Dataset:
        """
        Preprocesses the data & returns a HuggingFace dataset.

        Returns:
            Dataset: The HuggingFace dataset.
        """

        data = [asdict(sample) for sample in self._raw_data]
        dataset = Dataset.from_list(data)
        if self._scope == Scope.TRAINING:
            template_mapping = self._templates.format_train
        else:
            template_mapping = self._templates.format_infer
        dataset = dataset.map(self.clean)
        dataset = dataset.map(template_mapping, remove_columns=dataset.column_names)

        return dataset