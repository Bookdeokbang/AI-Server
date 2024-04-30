from fastapi import APIRouter, HTTPException
from typing import List
from models import AIModel
from bert import bert_classification
from roberta import roberta_classification
from albert import albert_classification
from google.cloud import vision
from models import OCRResponse
from spanbert import spanbert_classification

predict_router = APIRouter()
ocr_router = APIRouter()

client_options = {"api_endpoint": "eu-vision.googleapis.com"}
client = vision.ImageAnnotatorClient(client_options=client_options)

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
    return response

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