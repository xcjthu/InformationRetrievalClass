import json
import os
from torch.utils.data import Dataset


class SentDataset(Dataset):
	def __init__(self, config, mode, encoding="utf8", *args, **params):
		self.config = config
		self.mode = mode

		self.data_path = config.get("data", "%s_data_path" % mode)
		fin = open(self.data_path, 'r')
		self.data = [Tree(line) for line in fin]
		max_len = 0
		for d in self.data:
			max_len = max(max_len, len(d.parse()))
		print(max_len)

	def __getitem__(self, item):
		return self.data[item]

	def __len__(self):
		return len(self.data)


class Tree:
	def __init__(self, txt):
		self.childs = []
		txt = txt.strip()
		self.score = int(txt[1])
		
		chd_txt = []
		chd_level = 0
		for it in txt[3:-1]:
			if it == "(":
				if chd_level == 0:
					chd_txt.append("(")
				else:
					chd_txt[-1] += "("
				chd_level += 1
			elif it == ")":
				chd_level -= 1
				chd_txt[-1] += ")"
			elif chd_level != 0:
				chd_txt[-1] += it
		for subt in chd_txt:
			self.childs.append( Tree(subt) )
		if len(self.childs) == 0:
			self.text = txt[3:-1]
	def parse(self):
		if len(self.childs) == 0:
			return [self.text]
		else:
			ret = []
			for chd in self.childs:
				ret.extend(chd.parse())
			return ret


