#불러오기
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import fill
import numpy as np

idg = ImageDataGenerator(
    rescale = 1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest'
    
)

# trainhorse = idg.flow_from_directory('D:\study_data\_data\image\horse-or-human\horse-or-human/',
#                                      target_size=(300,300),
#                                      batch_size=3000,
#                                      class_mode='binary',
#                                      shuffle=True,
#                                     )
# trainhuman = idg.flow_from_directory('D:\study_data\_data\image\horse-or-human\horse-or-human\humans',
#                                      target_size=
#                                      (300,300),
#                                      batch_size=10,
#                                      class_mode='binary',
#                                      shuffle=True,
#                                      )



# np.save('D:\study_data\_save\_npy/horsman-x.npy',arr = trainhorse[0][0])
# np.save('D:\study_data\_save\_npy/horsman-y.npy',arr = trainhorse[0][1])

x = np.load('D:\study_data\_save\_npy/horsman-x.npy')
y = np.load('D:\study_data\_save\_npy/horsman-y.npy')

# print(trainhorse[1]) ValueError: Asked to retrieve element 1, but the Sequence has length 1
# 통짜데이터확인
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.6, random_state=66)
print(x.shape, y.shape)
from keras.models import Model, Input
from keras.layers import Dense, Conv2D, Flatten
# print(trainhorse[0][0].shape)
in1     = Input(shape=(300,300,3))
con2d1  = Conv2D(32, (4,4), activation='relu')(in1)
con2d2  = Conv2D(64, (4,4), activation='relu')(con2d1)
flat1   = Flatten()(con2d2)
dens1   = Dense(1, activation='sigmoid')(flat1)
model = Model(inputs = in1, outputs = dens1)
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics='accuracy')
hist = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
 
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('wj')
plt.ylabel('af')
plt.legend(loc ='lower center')
plt.title('dkjw')
plt.grid()
plt.gray()
plt.show()


# d?safaskdfalskjdfm