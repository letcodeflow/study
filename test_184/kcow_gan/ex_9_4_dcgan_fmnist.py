from keras.datasets import fashion_mnist
import numpy as np
from PIL import Image
import math
import os

import keras.backend as K
import tensorflow as tf

from keras import models, layers, optimizer_v2

def mse_4d(y_true, y_pred): #케라스용
    return K.mean(K.square(y_pred - y_true), axis=(1,2,3))

def mse_4d_tf(y_true, y_pred): #최적화 계산함수 텐서용
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=(1,2,3))


class GAN():
    def __init__(self, input_dim=64):
        self.input_dim = input_dim

        self.generator = self.GENERATOR()
        self.discriminator = self.DISCRIMINATOR()
        self.GD = self.make_GD()

    def GENERATOR(self):
        input_dim = self.input_dim

        model = models.Sequential()
        model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        model.add(layers.Dense(128*7*7,activation='tanh'))
        model.add(layers.BatchNormalization())
        model.add(layers.Reshape((7,7,128), input_shape=(128*7*7,)))
        model.add(layers.UpSampling2D(size=(2,2)))
        model.add(layers.Conv2D(64, (5,5), padding='same', activation='tanh'))
        model.add(layers.UpSampling2D(size=(2,2)))
        model.add(layers.Conv2D(1, (5,5), padding='same', activation='tanh'))

        model.summary()

        model.compile(loss=mse_4d_tf, optimizer='SGD')

        return model

    def DISCRIMINATOR(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64,(5,5), padding='same', activation='tanh', input_shape=(28,28,1)))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Conv2D(128,(5,5), activation='tanh'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.summary()

        d_optim = optimizer_v2.adam.Adam(lr = 0.0005)
        model.compile(loss = 'binary_crossentropy', optimizer=d_optim)

        return model

    def make_GD(self):
        G, D = self.generator, self.discriminator
        GD = models.Sequential()
        GD.add(G)
        GD.add(D)
        D.trainable = False
        g_optim = optimizer_v2.adam.Adam(lr=0.0005)
        GD.compile(loss = 'binary_crossentropy', optimizer=g_optim)
        D.trainable = True
        return GD

    def get_z(self, ln):
        input_dim = self.input_dim
        return np.random.uniform(-1,1,(ln, input_dim))

    def train_both(self, x):
        ln = x.shape[0]
        #first trial for training discriminator
        z = self.get_z(ln)
        w = self.generator.predict(z, verbose=0)
        xw = np.concatenate((x,w))
        y2 = ([[1]* ln + [0]* ln])
        #y2 = y.resh
        #y2 = y2.reshae(32,1)

        valid = np.ones((16,1))
        fake = np.zeros((16,1))
        y2 = np.concatenate((valid, fake))

        d_loss = self.discriminator.train_on_batch(xw,y2)

        #second trial for trainning gen
        z = self.get_z(ln)
        self.discriminator.trainable = False
        g_loss = self.GD.train_on_batch(z, valid)
        self.discriminator.trainable = True

        return d_loss, g_loss



def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]), dtype = generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index%width
        image[i * shape[0]:(i +1)* shape[0], j*shape[1]:(j+1)*shape[1]] = img[0,:,:]
    return image

def get_x(X_train, index, BATCH_SIZE):
    return X_train[index* BATCH_SIZE: (index+1)*BATCH_SIZE]

def save_images(generated_images, output_fold, epoch, index):
    generated_images = generated_images.reshape((generated_images.shape[0], 1) + generated_images.shape[1:3])
    image = combine_images(generated_images)
    image = image*127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(output_fold + '/' + str(epoch)+ '_'+ str(index)+'.png')

def load_data(n_train):
    (X_train, y_train), (_,_) = fashion_mnist.load_data()
    return X_train[:n_train]

def train(BATCH_SIZE, epochs, output_fold, input_dim, n_train):
    os.makedirs(output_fold, exist_ok=True)
    print('outpufol is ', output_fold)

    X_train = load_data(n_train)

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0],) + X_train.shape[1:]+(1,))

    gan = GAN(input_dim)

    d_loss_ll, g_loss_ll = [], []
    for epoch in range(epochs):
        print('Epoch is ', epoch)
        print('num of batchs', int(X_train.shape[0]/BATCH_SIZE))

        d_loss_l, g_loss_l = [], []
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            x = get_x(X_train, index, BATCH_SIZE)
            d_loss, g_loss = gan.train_both(x)

            d_loss_l.append(d_loss)
            g_loss_l.append(g_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            z = gan.get_z(x.shape[0])
            w = gan.generator.predict(z, verbose=0)
            save_images(w, output_fold, epoch, 0)

        d_loss_ll.append(d_loss_l)
        g_loss_ll.append(g_loss_l)

    gan.generator.save_weights(output_fold + '/'+'generator', True)
    gan.discriminator.save_weights(output_fold + '/'+'discriminator', True)

    np.savetxt(output_fold + '/' + 'd_loss' , d_loss_ll)
    np.savetxt(output_fold + '/' + 'g_loss' , g_loss_ll)

import argparse

def main():
    train(16,5000,'D:/OneDrive - 한국방송통신대학교/1_total_beat/study/test_184/kcow_gan/fhsin-mnist', 10, 32)

if __name__=='__main__':
    main()
