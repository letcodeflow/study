import config
from keras_preprocessing.image import img_to_array, load_img
from keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os
import keras

curd = os.path.dirname(os.path.abspath(__file__))+'\\'

#파이썬 입력 인자 파싱
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', default=curd+'dataset/images',
help='path to input image/text file of image filenames')
args = vars(ap.parse_args())

#입력폴더에서 파일명 획득
input_path = args['input']
imagePaths = []
for path in os.listdir(input_path):
    if os.path.isfile(os.path.join(input_path, path)):
        imagePaths.append(input_path+'\\'+path)

#학습모델 로딩
print('info loadin object detector')
model = load_model(config.MODEL_PATH)

#이미지 파일을 로딩해 학습모델에 입력 프레딕션
for imagePath in imagePaths:
    #로딩후 정규화
    #0축 차원 확장해 학습모델과 일치
    image = keras.utils.load_img(imagePaths, target_size = (224,224))
    image = img_to_array(image)/255.
    image = np.expand_dims(image, axis=0) #1,224,224,3

    #예측후 첫번째 출력 bbox값
    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds

    #이미지 로딩해 width 리사이즈 후 이미지폭너비 획득
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h,w) = image.shape[:2]

    #bbox 는 0-1사이이므로 스케일
    startX = int(startX*w)
    startY = int(startY*h)
    endX = int(endX*w)
    endY = int(endY*h)

    cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0),2)

    cv2.imshow('output', image)
    cv2.waitkey(0)
