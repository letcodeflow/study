from keras.preprocessing.text import Tokenizer

test1 = '나는 밥을 먹는다 디디디세네레란 먹는다 가덕더넌lkdsja lsdjjl sd jflwjasl 나는 먹는다 먹는다 먹는다'
test2 = '로드 암쏘롱 구구 지나닥 타타 루나타 타타 바피나 타타'

token = Tokenizer()
token.fit_on_texts([test1, test2]) #리스트 형태, 여러개 받을 수 있음, 인덱스 생성됨

print(token.word_index) #반복은 하나로 처리, 가장 많은 것을 앞으로
# {'먹는다': 1, '나는': 2, '밥을': 3, '디디디세네레란': 4, '가덕더넌lkdsja': 5, 'lsdjjl': 6, 'sd': 7, 'jflwjasl': 8}

x = token.texts_to_sequences([test1, test2])
print(x)
# [[2, 3, 1, 4, 1, 5, 6, 7, 8, 2, 1, 1, 1]]

from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
x_new = x[0] + x[1]
# print(x_new.shape)
x_new = to_categorical(x_new)
print(x_new)
print(x_new.shape)

# [[[0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]]]
# (1, 13, 9)





