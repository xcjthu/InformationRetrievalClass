import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.accuracy_init import init_accuracy_function
from .Encoder import TextLSTM, TextCNN

class SentClassify(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(SentClassify, self).__init__()

        self.sentiment_num = config.getint("model", "sentiment_num")
        self.encoder_name = config.getint("model", "encoder_name")
        self.emb_size = config.getint("model", "emb_size")
        self.hidden = config.getint("model", "hidden_size")

        if self.encoder_name == 'CNN':
            self.encoder = TextCNN(config, self.emb_size)
        elif self.encoder_name == "LSTM":
            self.encoder = TextLSTM(config, self.emb_size)
        self.out = nn.Linear(self.hidden, self.sentiment_num)

        self.loss = nn.CrossEntropyLoss()
        self.accuracy_function = init_accuracy_function(config, *args, **params)


    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['sent']
        label = data['labels']

        sent = self.encoder(x)
        out = self.out(sent)
        loss = self.loss(out, label)
        acc_result = self.accuracy_function(y, label, config, acc_result)
        return {"loss": loss, "acc_result": acc_result}
        

