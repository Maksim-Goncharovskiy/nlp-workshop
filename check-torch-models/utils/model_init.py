import torch
from transformers import AutoModel, AutoTokenizer


class BertMultiLabel(torch.nn.Module):
    def __init__(self, bert, n_classes):
        super(BertMultiLabel, self).__init__()
        self.bert = bert
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, ids, mask):
        _, output = self.bert(ids, attention_mask=mask)
        output = self.dropout(output)
        output = self.fc(output)
        return output


class Load:
    def __init__(self, model_name, model_hagging_face_name):

        bert = AutoModel.from_pretrained(model_hagging_face_name, return_dict=False)

        self.tokenizer = AutoTokenizer.from_pretrained(model_hagging_face_name)
        self.model = BertMultiLabel(bert, 50)
        self.model.load_state_dict(torch.load(f"../models/{model_name}", weights_only=True))