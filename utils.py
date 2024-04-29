import json

def load_labels_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:
        label_classes = json.load(json_file)
    print(type(label_classes))
    return label_classes