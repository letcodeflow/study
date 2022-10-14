from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

# model = Sequential()
# # model.add(Dense(units=10, input_shape=(10,10,3)))
# # model.add(Conv2D(filters=10, kernel_size=(3,3), input_shape=(10,10,3)))
# # 8,8,10
# # model.summary()
# # model.add(Dense(units=10, input_shape=(3,)))
# model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))
# # (None, 4, 4, 10)          50 출력
# model.add(Conv2D(7, (2,2), activation='relu'))
# # (None, 3, 3, 7)           287 출력
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax')) 
# 5by5 이미지에서 10칼럼으로 줄여서 그중에 제일큰값에 1을 준다(softmax)
# model.summary()

model = Sequential()
model.add(Conv2D(filters=2, kernel_size=(2,2), input_shape=(8,8,3)))
# none,6,6,10 370
# 데이터셋이 따로없으므로 데이터셋숫자는 none 8by8을 3by3으로 세면 6by6 상자가 나오므로 6,6 여기에 노드를 더한 값에 필터를 곱하면 370
model.add(Conv2D(7, (2,2), activation='relu'))
# none,5,5,7 = 1820
model.add(Flatten())
#  (None, 252)  이유?
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) 
# 5by5 이미지에서 10칼럼으로 줄여서 그중에 제일큰값에 1을 준다(softmax)
model.summary()
