#패키지 불러오기
#파라미터 설정
#모델 구현
#데이터 불러오기
#학습및 검증
#결과 분석

from keras import layers, models
class ANN_models_class(models.Model):
    def __init__(self, Nin, Nh, Nout):
        #prepare network layers and activate function
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        #connect networdk elements
        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x,y)
        self.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import np_utils

def Data_func():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W*H)
    X_test = X_test.reshape(-1, W*H)

    X_train = X_train/255.
    X_test = X_test/255.

    return (X_train, Y_train), (X_test, Y_test)


def main():
    Nin = 784
    Nh = 100
    number_of_class = 10
    Nout = number_of_class
    np.set_printoptions(linewidth=200)

    model = ANN_models_class(Nin, Nh,Nout)
    (X_train, Y_train), (X_test, Y_test)  = Data_func()
    print(X_train[0])

    history = model.fit(X_train, Y_train, epochs=15)
    pred = model.predict(X_test)
    print(pred[0])
    print(np.argmax(pred[0]))
    # print(history)

if __name__ == '__main__':
    main()
