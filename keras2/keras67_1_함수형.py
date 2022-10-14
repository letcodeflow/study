from xml.etree.ElementInclude import include
from keras.applications import vgg16,vgg19,resnet,resnet_v2,densenet,inception_resnet_v2,inception_v3,mobilenet,mobilenet_v2, efficientnet,xception
from keras.datasets import cifar10
from keras.optimizers import adam_v2
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D,Input

#1. data
# (x_train,y_train),(x_test,y_test) = cifar10.load_data()

models = [vgg16.VGG16,vgg19.VGG19,
# resnet.ResNet,
# densenet.DenseNet,
# inception_resnet_v2.InceptionResNetV2,
mobilenet.MobileNet,
# efficientnet.EfficientNet,
# xception.Xception
]
trainable = [0,1]
for reage in models:
    input = Input(shape=(32,32,3))
    func = reage(include_top=False)(input)
    func = GlobalAveragePooling2D()(func)
    func = Dense(100)(func)
    output = Dense(10,activation='softmax')(func)
    model = Model(inputs=input,outputs=output)
    for i in trainable:
        model.trainable=i
        print(reage.__name__,len(model.trainable_weights),'non',len(model.non_trainable_weights),sep='/')

# VGG16/0/non/30
# VGG16/30/non/0
# VGG19/0/non/36
# VGG19/36/non/0
# WARNING:tensorflow:`input_shape` is undefined or non-square, or `rows` is not in [128, 160, 192, 224]. Weights for input shape (224, 224) will be loaded as the default.
# MobileNet/0/non/139
# MobileNet/85/non/54




# def build_model(drop=0.5,optimizer=adam_v2.Adam,activation='relu',learning_rate=0.001,node1=128,node2=64):
    
