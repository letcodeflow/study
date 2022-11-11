import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import tensorflow as tf
y_true = np.array([100.,2.])
y_pred = np.array([200,102])
# y_true = np.array([100,200])
# y_pred = np.array([200,300])

mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print(mae,'\n', mape)
y_true = np.array([100,200])
y_pred = np.array([200,300])

mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print(mae,'\n', mape)

# import keras
# mape_tf = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
# print(mape_tf.numpy())