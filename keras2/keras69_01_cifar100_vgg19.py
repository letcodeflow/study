# xception
# resnet50
# resnet101
# inceptionv3
# inceptionresnetv2
# densenet121
# mobilenetv2
# nasnetmoobile
# efficientnetb0

from turtle import shape
from keras.applications import vgg19,resnet,inception_v3,densenet,mobilenet,nasnet,efficientnet,xception
from keras.datasets import cifar100
from keras.models import Model
from keras.layers import Dense,Input,GlobalAveragePooling2D
(x_train,y_train),(x_test,y_test) = cifar100.load_data()
input = resnet.ResNet(include_top=False,weights='imagenet',input_shape=(32,32,3))
input.trainable=False
hidden = input.output
hidden = GlobalAveragePooling2D()(hidden)
hidden = Dense(100,activation='softmax')(hidden)
model = Model(inputs=input.input,outputs=hidden)
model.compile(loss = 'sparse_categorical_crossentropy',optimizer='adam')
model.fit(x_train,y_train)
import numpy as np
from sklearn.metrics import accuracy_score
print(y_test.shape,model.predict(x_test).shape)
print(accuracy_score(y_test,np.argmax(model.predict(x_test),axis=1)))

# trainable 
# False
#  Ture

# vgg19.VGG19
# 0.2367
# 0.01

# mobilenet
# 0.0312
# 0.0982

