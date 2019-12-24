import json

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

