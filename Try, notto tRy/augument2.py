from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import numpy as np
x_train = np.load('d:/study_data/_save/_npy/brain/keras48_brain_x_train.npy')
y_train = np.load('d:/study_data/_save/_npy/brain/keras48_brain_y_train.npy')
x_test = np.load('d:/study_data/_save/_npy/brain/keras48_brain_x_test.npy')
y_test = np.load('d:/study_data/_save/_npy/brain/keras48_brain_y_test.npy')

print(y_test)
print(y_train)



model = Sequential()
model.add(Conv2D(15,(2,2), input_shape=(100,100,1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=150, batch_size=500, validation_split=0.2,)

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print(y_predict)
y_predict = y_predict.round()
print(y_predict)