import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.accuracy_init import init_accuracy_function
from .Encoder import TextLSTM, TextCNN, CNN

class SentClassify(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(SentClassify, self).__init__()

        self.sentiment_num = config.getint("model", "sentiment_num")
        self.encoder_name = config.get("model", "encoder_name")
        self.word_num = config.getint('data', 'word_num')
        self.emb_size = config.getint("model", "emb_size")
        self.hidden = config.getint("model", "hidden_size")
        self.dropout = config.getfloat("model", "dropout")

        self.embs = nn.Embedding(self.word_num, self.emb_size)
        # wordemb = np.load(config.get('data', 'wordvec_path'))
        # self.embs.weight.data.copy_(torch.from_numpy(wordemb))
        # self.embs.weight.requires_grad = False

        if self.encoder_name == 'CNN':
            # self.encoder = TextCNN(config, self.emb_size)
            self.encoder = CNN(config, self.emb_size)
        elif self.encoder_name == "LSTM":
            self.encoder = TextLSTM(config, self.emb_size)
        self.out = nn.Linear(self.hidden, self.sentiment_num)

        self.loss = nn.CrossEntropyLoss()
        self.accuracy_function = init_accuracy_function(config, *args, **params)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['sent']
        label = data['labels']

        x = self.embs(x)
        if mode == 'train':
            x = self.dropout(x)
        sent = self.encoder(x)
        out = self.out(sent)
        loss = self.loss(out, label)
        acc_result = self.accuracy_function(out, label, config, acc_result)
        return {"loss": loss, "acc_result": acc_result}
        

