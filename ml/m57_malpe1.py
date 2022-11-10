import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import tensorflow as tf
y_true = np.array([0.,2.,3.])
y_pred = np.array([2.,4.,2.])

mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print(mae,'\n', mape)

import keras
mape_tf = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
print(mape_tf.numpy())