#image 에 본인사진 넣고 predictfrom keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rescale = 1./255
)
me = datagen.flow_from_directory('C:/Users/aiapalm\Downloads\me/',target_size=(50,50),
                                     batch_size=3000000000,
                                     class_mode='binary',
                                     shuffle=True,)



# train = datagen.flow_from_directory('D:\study_data\_data\image/MENWOMEN/train/',target_size=(300,300),
#                                      batch_size=3000000000,
#                                      class_mode='binary',
#                                      shuffle=True,)
# test = datagen.flow_from_directory('D:\study_data\_data\image\MENWOMEN/test/',target_size=(300,300),
#                                      batch_size=300000000000,
#                                      class_mode='binary',
#                                      shuffle=True,)

# np.save('D:\study_data\_save\_npy\keras47_mwtrain_x', arr = train[0][0])
# np.save('D:\study_data\_save\_npy\keras47_mwtrain_y',arr = train[0][1])
# np.save('D:\study_data\_save\_npy\keras47_mwtest_x',arr = train[0][0])
# np.save('D:\study_data\_save\_npy\keras47_mwtest_y.npy',arr = train[0][1])

x_train = np.load('D:\study_data\_save\_npy\keras47_mwtrain_x.npy')
y_train = np.load('D:\study_data\_save\_npy\keras47_mwtrain_y.npy')
x_test = np.load('D:\study_data\_save\_npy\keras47_mwtest_x.npy')
y_test = np.load('D:\study_data\_save\_npy\keras47_mwtest_y.npy')
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) 
from keras.models import Model, Input
from keras.layers import Dense, Conv2D, Flatten
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
in1     = Input(shape=(50,50,3))
Con2d1  = Conv2D(2,(38,38), activation='relu')(in1)
Con2d2  = Conv2D(2,(10,10), activation='relu')(Con2d1)
flat1   = Flatten()(Con2d2)
dens1   = Dense(1, activation='sigmoid')(flat1)
model   = Model(inputs = in1, outputs = dens1)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

loss = model.evaluate(x_test, y_test)
pred = model.predict(me[0][0])

pred = np.argmax(pred, axis=1)

print(loss)
print(pred)