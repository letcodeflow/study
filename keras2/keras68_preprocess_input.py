from keras.applications.resnet import ResNet50,preprocess_input,decode_predictions
from keras.preprocessing import image
from matplotlib.pyplot import axis
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'C:/Users/aiapalm/OneDrive - KNOU/beat/study/_data/dog/170px-Golden_retriever_eating_pigs_foot.jpg'

img = image.load_img(img_path,target_size=(224,224))
print(img)

x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

pred = model.predict(x)
print(decode_predictions(pred))




