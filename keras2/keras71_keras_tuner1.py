#몇가지 방식중 하이퍼밴드
import tensorflow as tf
import keras
import keras_tuner as kt
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
# from keras.optimizer_v2 import adam
from keras.optimizers import Adam
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255.

def get_model(hp):
    hp_uint1 = hp.Int('uints1', min_value=16, max_value=512, step=16)
    hp_uint2 = hp.Int('uints2', min_value=16, max_value=512, step=16)
    hp_uint3 = hp.Int('uints3', min_value=16, max_value=512, step=16)
    hp_uint4 = hp.Int('uints4', min_value=16, max_value=512, step=16)

    hp_drop1 = hp.Choice('dropout1', values=[0.0,0.2,0.3,0.4,0.5])
    hp_drop2 = hp.Choice('dropout2', values=[0.0,0.2,0.3,0.4,0.5])

    hp_lr = hp.Choice('lerning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(hp_uint1, activation='relu'))
    model.add(Dropout(hp_drop1))
    
    model.add(Dense(hp_uint2, activation='relu'))
    model.add(Dropout(hp_drop1))
    model.add(Dense(hp_uint3, activation='relu'))
    model.add(Dropout(hp_drop2))
    model.add(Dense(hp_uint4, activation='relu'))
    model.add(Dropout(hp_drop2))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=hp_lr), loss = 'sparse_categorical_crossentropy', metrics='accuracy')

    return model

kerastuner = kt.Hyperband(get_model, directory='my_dir', objective='val_accuracy', max_epochs=6, project_name='kerastuner-mnist')
kerastuner.search(x_train, y_train, validation_data=(x_test, y_test), epochs=5,)

best_hp = kerastuner.get_best_hyperparameters(num_trials=2)[0]
print(best_hp.get('uints1'))
print(best_hp.get('uints2'))
print(best_hp.get('uints3'))
print(best_hp.get('uints4'))
print(best_hp.get('dropout1'))
print(best_hp.get('dropout2'))
print(best_hp.get('lerning_rate'))




