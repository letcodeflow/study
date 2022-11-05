#pixel마다의 클래스 semantic image segmentation - 의미있는 것끼리 묶는다 dense prediction - 픽셀마다의 레이블이 들어가기 때문에 어려움

import os
input_dir = 'D:/OneDrive - 한국방송통신대학교/data/oxford pets/images/images'
target_dir = 'D:/OneDrive - 한국방송통신대학교/data/oxford pets/annotations/annotations/trimaps'
img_size = (160,160)
num_classes = 3
batch_size = 32
input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith('jpg')])
# print(input_img_paths)
# ['D:/OneDrive - 한국방송통신대학교/data/oxford pets/images/images\\Abyssinian_1.jpg', 'D:/OneDrive - 한국방송통신대학교/data/oxford pets/images/images\\Abyssinian_10.jpg', 'D:/OneDrive - 한국방송통신대학교
target_img_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith('.png') and not fname.startswith('.')])
# print(target_img_paths)

from IPython.display import Image, display
from keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
display(Image(filename=input_img_paths[7]))
img = PIL.ImageOps.autocontrast(load_img(target_img_paths[7]))
display(img)
