import config
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import adam_v2
from keras_preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


print('info loading dataset..')
rows = open(config.ANNOTS_PATH).read().strip().split('\n')
# print(rows)
# 'image_0335.jpg,53,30,350,112', 'image_0336.jpg,55,31,351,119', '
data, targets, filenames = [],[],[]
#이미지 데이터리스트, bbox좌표, 각이미지 파일명

for row in rows:
    row = row.split(',')
    (filename, startX, startY, endX, endY) = row

    imagePath = os.path.sep.join([config.IMAGES_PATH, filename])

    img_array=np.fromfile(imagePath,np.uint8)
    image=cv2.imdecode(img_array,cv2.IMREAD_COLOR)

    # image = cv2.imread(imagePath)
    (h,w) = image.shape[:2]
    print(image.shape)
    #경계박스 정규화
    startX = float(startX)/w
    startY = float(startY)/h
    endX = float(endX)/w
    endY = float(endY)/h

    image = load_img(image, target_size=(224,224))
    # print(image.shape)
    image = img_to_array(image)
    # print(image.shape)
    # print(image)

    #이미지를 데이터배열에 추가, 타겟에 bbox 추가, 파일명 리스트 추가
    data.append(image)
    targets.append((startX, startY, endX, endY))
    filenames.append(filename)

#데이터를 넘파이형식으로 변환 255로 나누어 정규화
data = np.array(data, dtype='float32')/255.
targets = np.array(targets, dtype='float32')

split = train_test_split(data, targets, filenames, test_size=.1, random_state=22)

(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

print('info savegin testing filenames')
f = open(config.TEST_FILENAMES, 'w')
f.write('\n'.join(testFilenames))
f.close()

#마지막 층 제외하고 로딩
vgg = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

#학습 데이터가 사전학습데이터와 유사므르ㅗ 비학습
vgg.trainable = False

faltten = vgg.output
flatten = Flatten()(flatten)

#좌표값 예측
bboxHead = Dense(128, activation='relu')(flatten)
bboxHead = Dense(64, activation='relu')(bboxHead)
bboxHead = Dense(32, activation='relu')(bboxHead)
bboxHead = Dense(4, activation='sigmoid')(bboxHead)

#vgg.input.shape. = none, 224, 224,3
# output = bboxhead.shape = none, 4
model =Model(inputs =vgg.input, outputs = bboxHead)

opt = adam_v2.Adam(lr=config.INIT_LR)
model.compile(loss='mse', optimize=opt)
print(model.summary())

print('info training bbox regressor')
H = model.fit(
    trainImages, trainTargets,
    validation_data = (testImages, testTargets),
    batch_size = config.BATCH_SIZE,
    epochs = config.NUM_EPOCHS,
    verbose=1
)

print('saving model')
model.save(config.MODEL_PATH, save_format='h5')


N = config.NUM_EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0,N), H.history['val_loss'], label='val_loss')
plt.title('bbox regression losso n train seet')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc = 'lower left')
plt.savefig(config.PLOT_PATH)
