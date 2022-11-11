import os
import pickle
import numpy as np
from tqdm.notebook import tqdm

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, Conv1D
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(118,))
se1 = Embedding(400, 256, mask_zero=True)(inputs2)
se2 = Dense(256, activation='relu')(se1)
se3 = Dropout(0.4)(se1)

# decoder model
decoder1 = add([fe2, se3])
# decoder2 = LSTM(128)(decoder1)
decoder2 = Conv1D(256,3)(decoder1)
decoder3 = Dense(256, activation='relu')(decoder2)
outputs = Dense(400, activation='softmax')(decoder3)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.summary()