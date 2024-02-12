from enum import Enum
from pathlib import Path


class Scope(Enum):
    """The scope of a variable."""

    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "test"
    INFERENCE = "inference"


CACHE_DIR = Path.home() / ".cache" / "llms"