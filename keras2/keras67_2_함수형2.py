from keras.applications import vgg16,vgg19,resnet,resnet_v2,densenet,inception_resnet_v2,inception_v3,mobilenet,mobilenet_v2, efficientnet,xception
from keras.datasets import cifar10
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D,Input
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print(x_train.shape)
base_model = vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(32,32,3))
# print(base_model.layers[50])
# base_model.summary()

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(100,activation='relu')(x)

output = Dense(10,activation='softmax')(x)

model = Model(inputs= base_model.input,outputs=output)

# #1.
# for layer in base_model.layers: #base_model.layers[3]
#     layer.trainable=False
#     Total params: 54,353,206
# Trainable params: 16,470
# Non-trainable params: 54,336,736
#2.
base_model.trainable = False
# Total params: 54,353,206
# Trainable params: 16,470
# Non-trainable params: 54,336,736
model.summary()


model.compile(loss = 'sparse_categorical_crossentropy',optimizer=adam.Adam())
model.fit(x_train,y_train,validation_split=0.2,batch_size=128)
from sklearn.metrics import accuracy_score
import numpy as np
print(accuracy_score(y_test,np.argmax(model.predict(x_test),axis=1)))
