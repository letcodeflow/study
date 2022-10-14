#하이퍼파라미터 추가- 러닝레이트, 노드
import numpy as np
import pandas as pd
from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout,LSTM
import keras
import tensorflow as tf
print(tf.__version__)
#1. data
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(-1,28*28).astype('float32')/255.
x_test = x_test.reshape(-1,28*28).astype('float32')/255.

from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2.model
def build_model(drop='drop',optimizer='adam',activation='relu',node='node',lr='lr',model='model'):
    inputs = Input(shape=(28*28,),name='input')
    x = model(node, activation=activation,name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation,name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation,name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax',name='outputs')(x)

    model = Model(inputs=inputs,outputs=outputs)
    # optimizer=optimizer(lr)
    model.compile(optimizer=optimizer,metrics=['accuracy'], loss = 'sparse_categorical_crossentropy')

    return model

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizer = ['adam','rmsprop','adadelta']
    dropout = [0.3,0.4,0.5]
    lr = [0.01,0.002,0.05]
    node = [10,20]
    epochs = [1,2,3]
    activation = ['relu','linear','sigmoid','elu','selu']
    model =[Dense]
    return {'batch_size':batchs, 'optimizer':optimizer,'drop':dropout,'activation':activation, 'lr':lr,'drop':dropout,'node':node,'epochs':epochs,'model':model}

hyperparameters = create_hyperparameter()
# print(hyperparameters)
# {'batch_size': ([100, 200, 300, 400, 500],), 'optimizer': ['adam', 'rmsprop', 'adadelta'], 'drop': [0.3, 0.4, 0.5], 'activation': ['relu', 'linear', 'sigmoid', 'elu', 'selu']}

from sklearn.model_selection import GridSearchCV, cross_val_score,RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
keras_model = KerasClassifier(build_fn = build_model,verbose=1)
# model = GridSearchCV(keras_model,param_grid=hyperparameters,cv=2)
model = RandomizedSearchCV(keras_model,param_distributions=hyperparameters,cv=2,n_iter=2)
import time
start = time.time()
model.fit(x_train,y_train,epochs=1,validation_split=0.3)
end = time.time()

# loss, acc = model.evaluate(x_test,y_test)
print('time',end-start)
print('best-params',model.best_params_) #best-params {'optimizer': 'rmsprop', 'drop': 0.5, 'batch_size': 400, 'activation': 'relu'}
print('best-esti',model.best_estimator_)
print('best-bestscore',model.best_score_)
print('score',model.score)

from sklearn.metrics import accuracy_score
pred = model.predict(x_test)
print('acc',accuracy_score(y_test,pred))
