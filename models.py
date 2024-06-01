from enum import Enum
from pydantic import BaseModel


class OCRResponse(BaseModel):
    sentence: str


class AIModel(str, Enum):
    BERT = "bert"
    ROBERTA = "roberta"
    ALBERT = "albert"
    SPANBERT = "spanbert"


class ModelInfo(BaseModel):
    status: str
    model_name: str
    num_parameters: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    max_position_embeddings: int
    type_vocab_size: int
    initializer_range: float
    layer_norm_eps: float
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int