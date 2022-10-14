import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import Multi30k
# from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
print(torch.__version__)
from torchtext.legacy.data import Field, BucketIterator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy
import spacy.cli
spacy.cli.download('de_core_news_sm')
spacy.cli.download('en_core_web_sm')
import numpy as np
import random
import math
import time
from torchtext.data.metrics import bleu_score

SEED = 1000
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic =True
torch.backends.cudnn.benchmark = False

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

    
....s