import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from utils import load_labels_from_json

label_classes = load_labels_from_json('data/label.json')

def choice_roberta_model():
    model_path = "model/RoBERTa.pth"
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=53)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict.pop("roberta.embeddings.position_ids", None)
    model.load_state_dict(state_dict)
    model.eval()

    return tokenizer, device, model

def roberta_classification(sentences):
    tokenizer, device, model = choice_roberta_model()
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)
        predicted_classes = predicted_classes.tolist()
        predicted_label = [label_classes[idx] for idx in predicted_classes]

    return probs.tolist(), predicted_label

