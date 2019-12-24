from .nlp.BasicBert import BasicBert
from .SentClassify import SentClassify
model_list = {
    "BasicBert": BasicBert,
    "SentClassify": SentClassify,
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
