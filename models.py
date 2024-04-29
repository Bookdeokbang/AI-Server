from enum import Enum
from pydantic import BaseModel

class OCRResponse(BaseModel):
    sentence: str

class AIModel(str, Enum):
    BERT = "bert"
    ROBERTA = "roberta"