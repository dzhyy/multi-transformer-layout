from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn 

class Word2Vector():
    def __init__(self, local_files='./experiment/buffer/bert-base-chinese') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(local_files, local_files_only=True)
        pretrain_model = AutoModelForMaskedLM.from_pretrained(local_files, local_files_only=True)
        self.model = nn.Sequential(*list(pretrain_model.children())[:-1])
        for p in self.model.parameters():
            p.requires_grad = False

    def process(self, batch_sentence:list):
        inputs = self.tokenizer(batch_sentence, padding=True, truncation=True, max_length=10, return_tensors="pt")
        outputs = self.model(inputs['input_ids']).last_hidden_state
        return outputs


'''a = Word2Vector()
batch_sentence = [
    '这是第一个句子',
    '这是第二个句子',
    '第三个句子不一样。'
]
b = a.process(batch_sentence) # [3,10,768]
'''