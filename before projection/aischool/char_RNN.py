from __future__ import absolute_import, division, print_function, unicode_literals
#파이썬2와 호환성

from absl import app
from sympy import sequence #자주씀 
import tensorflow as tf

#파이썬
import numpy as np
import os
import time

from yaml import SequenceNode

#input 과 input을 한글자씩 뒤로 밑 타겟 데이터 생성 하는 함수 생성
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]

    return input_text,target_text

data_dir = tf.keras.utilis.get_file('shakespeare.txt','https://drive.google.com/file/d/0BzNAHGvG5wxZZjcwMDc5YjctNGI4OC00YzAxLWJhYTctNGQ3ZDNmMmZlMjNh/view?usp=sharing&resourcekey=0-TtsP3DAmXlQHuogzn5YDQg')
batch_size = 64 #training 64, samplig 1
seq_length = 100 #traing 100, samplig 1
embedding_dim = 256 
hidden_size = 1024 #node 갯수
num_epochs = 10

#file 읽기
text = open(data_dir, 'rb').read().decode(encoding='utf-8')
#학습데이터에 포함된 모든 캐럭터를 나타내는 변수 vocab과 id를 부여해 딕트형태로 만든 char2idx 선언
vocab = sorted(set(text)) #캐릭터 갯수 set 받은 인수를 전부 분해해서 가져옴
vocab_size = len(vocab) #캐릭터 유니크 갯수
print('{} unique char'.format(vocab_size))
char2idx = {u: i for i , u in enumerate(vocab)} #딕트형태로 불러옴
idx2char = np.array(vocab) 

#학습데이터를 intㄹ 변환
text_as_int = np.array([char2idx[c] for c in text])

#split input target 함수 이용 인풋과 타겟데이터 사용
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)

#data 이용 섞거 
data = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)

#RNN
class RNN(tf.keras.Model):
    def __init__(self):
        super(RNN, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]) #임베딩 디멘션으로 바꿈
        self.hidden_layer_1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True, stateful=True, reccurent_initializer='glorot_uniform')
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        embedded_input = self.embedding_layer(x)
        features = self.hidden_layer_1(embedded_input)
        logits = self.output_layer(features)

        return logits

def sparse_cross_entropy(labels, logits):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
    #그라운드 트루스 라벨 값을 값을 원핫 인코딩하고 소프트맥스 해서 비교함

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(model, input, target):
    with tf.GradientTape() as tape:
        logits = model(input)
        loss = sparse_cross_entropy(target, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def generate_text(model, start_string):
    num_sampling = 4000 #생성할 글자 갯수

    #start string int 변환
    input_eval = [char2idx[s] for s in  start_string]
    input_eval = tf.expand_dims(input_eval,0) #테스트시 임의 1배치사이즈 

    #저장할 배열 초기화
    text_generated = []

    #낮으면 정확 높으면 다양
    temperature = 1.0

    #배치1
    model.reset_state()
    for i in range(num_sampling):
        predictions = model(input_eval)
        #불필요한 디멘션 삭제
        predictions = tf.squeeze(predictions, 0) #더미 디멘션 삭제 vocab_size 형태로 만듦
        #예측결과에 기반해 랜덤 샘플링하기 위해 categorical distribution
        predictions = predictions /temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        #argmax 샘플링은 정확도는 높지만 높은 확률만 취하다 보니 다음에 오는 글자가 정해짐, -> 다른 글자가 오도록 만들기
        #확률분포 중 하나로 가중치를 랜덤으로 주어 뽑힐 가능성을 섞는것

        #예측된 .캐릭터를 다음 인풋으로 활용
        input_eval = tf.expand_dims([predicted_id],0)
        #샘플링 결과를 text generateㅇ 배열에 추가
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

def main():

        
