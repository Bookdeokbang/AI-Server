import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from utils import load_labels_from_json
from openai import OpenAI
from settings import api_key
import glob
import os
import pandas as pd

label_classes = load_labels_from_json('data/label.json')
folder_path = 'data/pos'

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
        max_value, max_index = torch.max(probs, dim=-1)
        predicted_label = label_classes[max_index]

    return predicted_label, max_value.item()


def create_sentence(label,sentences):

    client = OpenAI(
    api_key = api_key
    )

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "너는 영어문장 생성기야."},
        {"role": "user", "content":f'{label}의 예시로는 이런것들이 있어, {sentences}'},
        {"role": "user", "content": f"{label}이랑 비슷한 영어 문장 하나만 만들어줘, 그냥 문장만 말해 다른거 말하지말고"},
      ]
    )
    return response.choices[0].message.content

def load_data_from_csv(folder_path):
    all_texts = []
    all_labels = []
    lists = {}
    # 파일 이름을 라벨로 사용
    for file_path in glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True):
        label = os.path.splitext(os.path.basename(file_path))[0]  # 파일 확장자 제거
        lists[f'list_{label}'] = []
        try:
            df = pd.read_csv(file_path)
            # 'Original Sentence' 열이 있는지 확인
            if 'Original Sentence' in df.columns:
                texts = df['Original Sentence'].tolist()
                all_texts.extend(texts)
                all_labels.append(label)
                lists[f'list_{label}'].extend(texts)
            else:
                print(f"'Original Sentence' column not found in {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return all_texts, all_labels, lists

all_texts, all_label, all_list = load_data_from_csv(folder_path)

def create_ten_sentence(label):
    lists = {}
    lists[f'list_{label}'] = []
    for i in range(10):
        sentence = create_sentence(label, all_list[f'list_{label}'])
        if sentence not in all_texts:
            all_texts.append(sentence)
            lists[f'list_{label}'].append(sentence)

    return lists

def choice_one_sentence(lists, label):
    persente = []
    predict_label = []
    for text in lists[f'list_{label}']:
        label, persent = roberta_classification(text)
        predict_label.append(label)
        persente.append(persent)

    return f"{predict_label[persente.index(max(persente))]}: {lists[f'list_{label}'][persente.index(max(persente))]}"