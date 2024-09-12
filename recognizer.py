import torch
from importlib import import_module

LABEL_KEY = {
    0: 'not_complaint',
    1: 'complaint',
}
MODEL_NAME = "ERNIE"

class ComplaintRecognizer:
    def __init__(self):
        self.label_key = LABEL_KEY
        self.model_name = MODEL_NAME

        model_module = import_module('models.' + self.model_name)
        self.config = model_module.Config('dataset')
        self.model = model_module.Model(self.config).to(self.config.device)
        self.model.load_state_dict(torch.load(self.config.save_path))
        self.model.eval()

    def build_predict_text(self, text):
        token = self.config.tokenizer.tokenize(text)
        token = ['[CLS]'] + token
        seq_len = len(token)
        mask = []
        token_ids = self.config.tokenizer.convert_tokens_to_ids(token)
        pad_size = self.config.pad_size
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        ids = torch.LongTensor([token_ids]).cuda()
        seq_len = torch.LongTensor([seq_len]).cuda()
        mask = torch.LongTensor([mask]).cuda()
        return ids, seq_len, mask

    def predict(self, text):
        data = self.build_predict_text(text)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            num = torch.argmax(outputs)
        return self.label_key[int(num)]
