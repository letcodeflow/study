from click import argument
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)




argument_size = 20
print(x_train.shape[0])

randidx = np.random.randint(x_train.shape[0], size = argument_size)

print(np.min(randidx), np.max(randidx))
print(type(randidx))
x_argumented = x_train[randidx].copy()
y_argumented = y_train[randidx].copy()
print(x_argumented.shape)
print(y_argumented.shape)
x_train = x_train.reshape(60000, 28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

x_argumented = x_argumented.reshape(x_argumented.shape[0], x_argumented.shape[1], x_argumented.shape[2],1)
import time
start_time = time.time()
train_datagen.flow(x_argumented, y_argumented,
                                  batch_size=argument_size,
                                  save_to_dir="c:/study_data/_temp/",
                                  shuffle=False,).next()[0]
end_time = time.time()-start_time
print(round(end_time, 3), '초')
print(x_argumented)
print(x_argumented.shape)

x_train = np.concatenate((x_train, x_argumented))
y_train = np.concatenate((y_train, y_argumented))

print(x_train.shape)
print(y_train.shape)

