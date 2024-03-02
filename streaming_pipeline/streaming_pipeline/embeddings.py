import logging
import traceback
from pathlib import Path
from typing import Optional, Union

import numpy as np
from transformers import AutoModel, AutoTokenizer

from streaming_pipeline.streaming_pipeline import constants
from streaming_pipeline.streaming_pipeline.base import SingletonMeta