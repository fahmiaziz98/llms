from dataclasses import dataclass
from typing import Dict, List, Union, Optional


"""
{
    'input': 'How can I lookup the business associated with a FEIN?',
    'output': "If the organization is a non-profit. You can search by EIN on Charity Navigator's website FOR FREE. https://www.charitynavigator.org/",
    'instruction': 'Utilize your financial knowledge, give your answer or opinion to the input question or subject . Answer format is not limited.'
}

refrensi:
    - https://github.com/shibing624/MedicalGPT/blob/main/supervised_finetuning.py
"""

@dataclass
class PromptTemplate:
    """A class representing a prompt template"""
    name: str
    instruction_template: str="{instruction}"
    chat_history_template: str="{chat_history}"
    question_template: str="{input}"
    answer_template: str="{output}"
    # Stop token, default is tokenizer.eos_token
    stop_str: Optional[str] = "</s>"

    @property
    def input_variables(self) -> List[str]:
        """Returns a list of input variables for the prompt template"""

        return ["instruction", "chat_history", "question", "answer"]

    @property
    def train_raw_template(self):
        """Returns the training prompt template format"""

        instruction = self.instruction_template
        chat_history = self.chat_history_template
        question = self.question_template
        answer = self.answer_template
        return f"{instruction}{chat_history}{question}{answer}{self.stop_str}"

    @property
    def infer_raw_template(self):
        """Returns the inference prompt template format"""

        instruction = self.instruction_template
        chat_history = self.chat_history_template
        question = self.question_template
        return f"{instruction}{chat_history}{question}{self.stop_str}"

    def format_train(self, sample: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        """Formats data sample for training sample"""

        prompt = self.train_raw_template.format(
            instruction_template=sample["instruction"],
            chat_history=sample.get("chat_history", ""),
            question=sample["question"],
            answer=sample["answer"],
        )
        return {"prompt": prompt, "payload": sample}

    def format_infer(self, sample: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        """Formats data sample for testing sample"""

        prompt = self.infer_raw_template.format(
            instruction_template=sample["instruction"],
            chat_history=sample.get("chat_history", ""),
            question=sample["question"],
        )
        return {"prompt": prompt, "payload": sample}


templates : Dict[str, PromptTemplate] = {}

def register_template(template: PromptTemplate):
    """Register a new conversation template."""
    templates[template.name] = template

def get_template(name: str) -> PromptTemplate:
    """Returns the template assigned to the given name"""
    return templates[name]

#### Register Templates ####
register_template(
    PromptTemplate(
        name="mistralai",
        instruction_template="[SYS]{instruction}[/SYS]",
        chat_history_template="summary: chat_history}",
        question_template="<s>[INST]{input}[/INST]",
        answer_template="{output}",
        stop_str="</s>"
    )
)