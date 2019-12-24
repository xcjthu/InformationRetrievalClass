import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autograd


class TextCNN(nn.Module):
    def __init__(self, config, data_size):
        super(TextCNN, self).__init__()

        self.data_size = data_size
        
        self.min_gram = config.getint("model", "min_gram")
        self.max_gram = config.getint("model", "max_gram")

        self.hidden = config.getint('model', 'hidden_size')

        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, config.getint('model', 'filters'), (a, self.data_size)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = (self.max_gram - self.min_gram + 1) * config.getint('model', 'filters')
        self.fc = nn.Linear(self.feature_len, self.hidden)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x = data
        
        batch_size = data.shape[0]
        x = x.view(x.shape[0], 1, -1, self.data_size)


        conv_out = []
        gram = self.min_gram
        
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(batch_size, -1)

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        y = self.fc(conv_out)
        # y = y / torch.norm(y, dim = 1, p = 2).unsqueeze(1)
        
        return y

class TextLSTM(nn.Module):
    def __init__(self, config, data_size):
        super(TextLSTM, self).__init__()
        self.data_size = data_size
        self.hidden_dim = config.getint("model", "hidden_size")

        self.lstm = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True,
                            num_layers=config.getint("model", "num_layers"))

        # self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, data):
        x = data
        
        batch_size = x.shape[0]

        x = x.view(batch_size, -1, self.data_size)

        lstm_out, self.hidden = self.lstm(x)
        
        # lstm_out = torch.max(lstm_out, dim=1)[0]

        y = torch.max(lstm_out, dim = 1)[0]

        return y

class CNN(nn.Module):
    def __init__(self, config, data_size):
        super(CNN, self).__init__()
        self.input_size = data_size
        self.hidden_size = config.getint('model', 'hidden_size')
        self.kernel_size = 3
        self.padding_size = 1
        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size)
        self.max_len = config.getint('data', 'max_len')
        self.pool = nn.MaxPool1d(self.max_len)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return x
