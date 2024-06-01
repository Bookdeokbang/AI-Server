import json

from fastapi import APIRouter, HTTPException
from typing import List

from google.oauth2 import service_account
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline, pipeline

from models import AIModel, ModelInfo
from bert import bert_classification
from roberta import roberta_classification, create_ten_sentence, choice_one_sentence, choice_roberta_model
from albert import albert_classification
from google.cloud import vision
from models import OCRResponse
from spanbert import spanbert_classification

predict_router = APIRouter()
ocr_router = APIRouter()

key_path = 'key.json'
# 서비스 계정 키를 사용하여 인증 설정
credentials = service_account.Credentials.from_service_account_file(key_path)
# Google Cloud Vision API 클라이언트 생성
client = vision.ImageAnnotatorClient(credentials=credentials)

@predict_router.post("/predict")
async def predict_endpoint(sentences: List[str], model: AIModel):
    response = {}
    for sentence in sentences:
        if model == AIModel.BERT:
            predicted_probs, predicted_classes = bert_classification(sentence)
        elif model == AIModel.ROBERTA:
            predicted_probs, predicted_classes = roberta_classification(sentences)
        elif model == AIModel.ALBERT:
            predicted_probs, predicted_classes = albert_classification(sentences)
        elif model == AIModel.SPANBERT:
            predicted_probs, predicted_classes = spanbert_classification(sentence)

        response[sentence] = predicted_classes
        response["label"] = predicted_probs
    return response

@predict_router.post("/generate")
async def generate_sentence(label: str):
    lists = create_ten_sentence(label)
    return choice_one_sentence(lists, label)


@ocr_router.post("/ocr", response_model=OCRResponse)
async def detect_text(image_url: str):
    try:
        image = vision.Image()
        image.source.image_uri = image_url

        response = client.text_detection(image=image)
        texts = response.text_annotations

        sentence = ' '.join(text.description for text in texts)

        return OCRResponse(sentence=sentence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@predict_router.post("/pos")
async def pos(text: str):
    file_path = "data/pos.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        pos_tags = json.load(file)

    pos_pipeline = pipeline("token-classification", model="QCRI/bert-base-multilingual-cased-pos-english")
    pos_results = pos_pipeline(text)

    response = {}
    for v in pos_results:
        word = v['word']
        entity = v['entity']

        # 키가 점(.)으로 시작하지 않는 경우에만 처리
        if not word.startswith('.'):
            pos_tag = pos_tags.get(entity, "Unknown")
            response[word] = pos_tag

    return response

@predict_router.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    tokenizer, _, model = choice_roberta_model()
    if model and tokenizer:
        status = "Model loaded successfully"
        model_name = model.config.name_or_path
        num_parameters = model.num_parameters()
        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        num_attention_heads = model.config.num_attention_heads
        vocab_size = model.config.vocab_size
        max_position_embeddings = model.config.max_position_embeddings
        type_vocab_size = model.config.type_vocab_size
        initializer_range = model.config.initializer_range
        layer_norm_eps = model.config.layer_norm_eps
        pad_token_id = model.config.pad_token_id
        bos_token_id = model.config.bos_token_id
        eos_token_id = model.config.eos_token_id
    else:
        status = "Model not loaded"
        model_name = ""
        num_parameters = 0
        num_layers = 0
        hidden_size = 0
        num_attention_heads = 0
        vocab_size = 0
        max_position_embeddings = 0
        type_vocab_size = 0
        initializer_range = 0.0
        layer_norm_eps = 0.0
        pad_token_id = 0
        bos_token_id = 0
        eos_token_id = 0

    return {
        "status": status,
        "model_name": model_name,
        "num_parameters": num_parameters,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "type_vocab_size": type_vocab_size,
        "initializer_range": initializer_range,
        "layer_norm_eps": layer_norm_eps,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id
    }