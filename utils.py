import json

def load_labels_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:
        label_classes = json.load(json_file)
    return label_classes