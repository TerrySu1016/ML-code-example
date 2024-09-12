import torch
from importlib import import_module

key = {
    0: 'not_complaint',
    1: 'complaint',
}

model_name = 'ERNIE'
x = import_module('models.' + model_name)
config = x.Config('dataset')
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path))
model.eval()

def build_predict_text(text):
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    pad_size = config.pad_size
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

def predict(text):
    data = build_predict_text(text)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
    return key[int(num)]

if __name__ == '__main__':
    input_list = [
        "新的那个兔子我拉人他那里点击后我这里不显示",
        "兔子我这里点击后不显示",
        "兔子我这里打不开",
        "新出的那个兔子我这里点击没有反应",
        "我这里看不见兔子",
        "我点了兔子后拉人但是我这里数量不上升啊"
    ]

    # f = open("preparation\\test.txt", encoding='UTF-8')
    # input_list = f.readlines()
    result = []
    for i in input_list:
        input = i.replace("\n", "")
        input = input.replace("兔子", "宝箱")
        output = predict(input)
        print(input)
        print(output)
