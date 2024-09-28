from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader, Dataset

from .text_dataset import TextDataset

from .get_data import get_data

from .model_init import Load


def calculate_accuracy(model, tokenizer, X_val, y_val):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    with torch.no_grad():
        text_dataset = TextDataset(X_val, y_val, tokenizer)
        data_loader = DataLoader(text_dataset, batch_size=len(text_dataset), shuffle=False)
        data = next(iter(data_loader))

        input_ids = data['input_ids'].to(device)  # токены
        attention_mask = data['attention_mask'].to(device)  # маски
        targets = data['labels']

        output = model(input_ids, attention_mask)

        predictions = (output.detach().numpy() >= 0.5).astype(int)
        y_true = targets.detach().numpy().astype(int)

    return accuracy_score(y_true, predictions)


def get_accuracy(model_name: str, model_hugging_face_name: str):

    X_val, y_val = get_data()

    load = Load(model_name, model_hugging_face_name)
    model = load.model
    tokenizer = load.tokenizer

    return calculate_accuracy(model, tokenizer, X_val, y_val)