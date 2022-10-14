#그냥 카테고리가 수치적으로 의미있다고 즉 결과값이 0.7이 나오면 1에 가까운 것처럼
#RNN 인풋쪽에서 상당히 영향력있는
from keras.preprocessing.text import Tokenizer
import numpy as np

#1.데이터
docs=['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요', '싶라']

#긍정 1, 부정0
labels = np.array([1,1,1,0])

#수치화
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, '싶라': 8}

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6, 7], [8]] 길이가 다 다르다
#CNN에서 padding을 씌우듯 가장 긴 데이터길이에 맞춰 짧은 데이터는 0을 빈자리에 채운다
#그러나 채워야할 빈자리를 너무 길게 만드는 outlier 즉 너무 긴 데이터는 잘라버린다

#0을 앞에 채우나 뒤에 채우나 상관없지만 LSTM의 특성상 뒤에 있는 값에 가중치가 부여된다
from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=9, truncating='pre')
#x에 패딩은 앞쪽, 맥스는 5글자로 하겠다
#trunccating 앞에서 자를거냐 뒤에서 자를거냐
print(pad_x)
# [[0 0 0 2 3]
#  [0 0 0 1 4]
#  [0 1 5 6 7]
#  [0 0 0 0 8]]
print(pad_x.shape) #(4, 5)
pad_x = pad_x.reshape(4,9,1)
word_size = len(token.word_index)
print('word_size는', word_size) #word_size는 8
print(np.unique(pad_x,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([11,  2,  1,  1,  1,  1,  1,  1,  1], dtype=int64))


#2.모델
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM, GRU, Reshape, Conv1D, Embedding #인풋레이어에서 많이 쓴다
#빈자리를 채우면 너무 많이 채우게 된다 데이터의 상관관계를 분석해 벡터화시키자


# in1 = Input(input_shape=(5,))
# em1 = Embedding(input_dim=5, output_dim=10, input_length=5)(in1) #통상 3차원 아웃풋이 나온다 ->LSTM
# #input_dim 단어의 갯수
# #param은 연산이 아니라 연산의 모양을 맞춰주는 것 inputdim*outputdim
# #inputlength는 길이를 몰라도 자동으로 잡아주겠다 none,none,10
# # em1 = Embedding(input_dim=4,output_dim=10) #ok
# # em1 = Embedding(4,10) #가능
# # em1 = Embedding(4,10,5) #에러
# # em1 = Embedding(4,10, input_length=5) #명시해줌 ok
# # 인풋렝스를 바꿔도 경고는 뜨지만 모델은 돌아간다 
# lst1 = LSTM(32)(em1)
# dens1 = Dense(1, activation='sigmoid')(lst1)
# model = Model(inputs=in1, outputs=dens1)
# model.summary()
print(type(pad_x), type(labels))
print(pad_x.shape, labels.shape)
model = Sequential()
# model.add(Embedding(4,10))
model.add(LSTM(32, input_shape=(9,1)))
model.add(Dense(123))
model.add (Dense(1, activation='sigmoid'))
model.summary()


#.3 데이터 붙이기 = 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x,labels, epochs=1, batch_size=1)

#4. 평가예측
acc = model.evaluate(pad_x, labels)[1]
print(acc)
#loss[0], acc[1] 이므로 acc를 뽑겠다


from keras.preprocessing.text import Tokenizer
#5predict 
pred = ['너무 참 싶라 영화에요 진짜 최고에요']
p = pred + docs
# token.fit_on_texts(p)
token.fit_on_texts(p)
print(token.word_index)
predid2 = token.texts_to_sequences(pred)
predid3 = pad_sequences(predid2, padding='pre',maxlen=9, truncating='pre')
print(predid3)
print(predid3.shape)
predict = model.predict(predid3)
predict = predict.round()
print(predict)
print(predict.shape)
# 프레딕트가 0.50654846 값이 나왔지만 어떻게 처리해야할지 모르겠ㄱ다
# {'참': 1, '너무': 2, '최고에요': 3, '영화에요': 4, '싶라': 5, '재밌어요': 6, '잘': 7, '만든': 8, '진짜': 9}
# [[0 0 0 2 1 5 4 9 3]]
# (1, 9)
# [[1.]]
# (1, 1)



