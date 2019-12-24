import json
import torch
import os
from formatter.Basic import BasicFormatter

class SentSentFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.max_len = config.getint("data", "max_len")
        self.word2id = json.load(open(config.get('data', 'word2id')))
        self.mode = mode
    
    def format_sent(self, sent):
        ret = []
        for w in sent:
            if w in self.word2id:
                ret.append(self.word2id[w])
            else:
                ret.append(self.word2id['<unk>'])
        while len(ret) < self.max_len:
            ret.append(self.word2id['<pad>'])
        ret = ret[:self.max_len]
        return ret
    
    def process(self, data, config, mode, *args, **params):
        inp = []
        labels = []
        for d in data:
            sent = d.parse()
            inp.append(self.format_sent(sent))
            labels.append(d.score)
        return {
            'sent': torch.LongTensor(inp),
            'labels': torch.LongTensor(labels),
        }

