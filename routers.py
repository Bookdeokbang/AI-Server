import json

from fastapi import APIRouter, HTTPException
from typing import List

from google.oauth2 import service_account
from sympy.geometry import entity
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline, pipeline

from models import AIModel
from bert import bert_classification
from roberta import roberta_classification, create_ten_sentence, choice_one_sentence
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

    response = []
    for v in pos_results:
        if v['entity'] in pos_tags.keys():
            entity = pos_tags[v['entity']]
        response.append([v['word'],entity])
    return response