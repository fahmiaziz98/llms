import hashlib
from tqdm import tqdm
from pydantic import BaseModel
from typing import Dict, Optional, List

from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars
from unstructured.staging.huggingface import chunk_by_attention_window
from unstructured.staging.huggingface import stage_for_transformers

from transformers import AutoModel, AutoTokenizer
from src.path import DATA_DIR
from src.logger import get_console_logger
from src.vector_db import get_qdrant_client, init_collection


NEWS_FILE = DATA_DIR / "financial_news.json"
QDRANT_COLLECION_NAME = "financial_news"
QDRANT_VECTOR_SIZE = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LOGGER = get_console_logger(__name__)
MODEL = AutoModel.from_pretrained(MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list] = []
    chunks: Optional[list] = []
    embeddings: Optional[list] = []

qdrant_client = get_qdrant_client()
qdrant_client = init_collection(
    qdrant_client=qdrant_client, 
    collection_name=QDRANT_COLLECION_NAME,
    vector_size=QDRANT_VECTOR_SIZE
)

def parse_document(_data: Dict) -> Document:
    pass