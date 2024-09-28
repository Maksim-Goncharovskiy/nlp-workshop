import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Класс для преобразования датасета к нужному формату"""
    def __init__(self, X, y, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.sentences = X['text'].tolist()
        self.labels = y['target_list'].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = self.sentences[idx]
        target_list = self.labels[idx]
        # токенизируем
        inputs = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True, # добавление спец-токенов, отвечающих за "начало предложения" [CLS] и "конец предложения" [SEP]
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False, # это для задачи вопросно-ответной системы, т.е. не для нас
            return_attention_mask=True,
            return_tensors='pt' # формат выдачи токенизатора, в нашем случае - torch тензор
        )

        # то что мы запихнем в модель
        return {
            'input_ids': inputs['input_ids'].flatten(), # это наши цифровые токены (т.е. для токена 'привет' будет какое-нибудь '105')
            'attention_mask': inputs['attention_mask'].flatten(), # это наши маски
            'labels': torch.tensor(target_list, dtype=torch.float)
        }