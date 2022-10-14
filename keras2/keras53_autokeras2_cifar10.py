import tensorflow as tf
print(tf.__version__)
import time
# from tensorflow.python.keras import dtensor

import autokeras as ak
# print(ak.__version__)

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)

#.컴파일 훈련
st = time.time()
model.fit(x_train,y_train,epochs=5)
end = time.time()
pred = model.predict(x_test)

r = model.evaluate(x_test,y_test)
print(r)
print(end-st)