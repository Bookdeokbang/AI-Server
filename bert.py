import torch
from transformers import BertModel, BertTokenizer
from utils import load_labels_from_json

label_classes = load_labels_from_json('data/label.json')

class MyBertModel(torch.nn.Module):
    def __init__(self):
        super(MyBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 53)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def choice_bert_model():
    model_path = "model/BERT.pth"
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MyBertModel()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict.pop("bert.embeddings.position_ids", None)
    model.load_state_dict(state_dict)
    model.eval()

    return tokenizer, device, model

def bert_classification(sentences):
    tokenizer, device, model = choice_bert_model()
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    inputs.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=-1)
        predicted_probs, predicted_classes = torch.max(probs, dim=-1)
        predicted_classes = predicted_classes.tolist()
        predicted_label = [label_classes[idx] for idx in predicted_classes]

    return predicted_probs.tolist(), predicted_label