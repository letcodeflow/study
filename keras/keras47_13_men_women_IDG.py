from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils import shuffle
datagen = ImageDataGenerator(
    rescale = 1./255
)
me = datagen.flow_from_directory('C:/Users/aiapalm\Downloads\me/')



train = datagen.flow_from_directory('D:\study_data\_data\image/MENWOMEN/train/',target_size=(100,100),
                                     batch_size=3000000000,
                                     class_mode='binary',
                                     shuffle=True,)
test = datagen.flow_from_directory('D:\study_data\_data\image\MENWOMEN/test/',target_size=(100,100),
                                     batch_size=300000000000,
                                     class_mode='binary',
                                     shuffle=True,)

np.save('D:\study_data\_save\_npy\keras47_mwtrain_x.', arr = train[0][0])
np.save('D:\study_data\_save\_npy\keras47_mwtrain_y',arr = train[0][1])
np.save('D:\study_data\_save\_npy\keras47_mwtest_x',arr = test[0][0])
np.save('D:\study_data\_save\_npy\keras47_mwtest_y.npy',arr = test[0][1])

print('move your ass')