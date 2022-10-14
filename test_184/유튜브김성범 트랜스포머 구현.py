import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import Multi30k
# from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
print(torch.__version__)
from torchtext.data import Field, BucketIterator
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