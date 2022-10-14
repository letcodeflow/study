#non linear 특성을 학습 sigmoid tanh relu
#keras subclassing
from statistics import mean
from numpy import average, gradient
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#타입변경
x_train, x_test = x_train.astype('float32'),x_test.astype('float32')
#flattening
x_train, x_test = x_train.reshape([-1,784]),x_test.reshape([-1,784])
#normalize
x_train, x_test = x_train/255., x_test/255.
#one-hot
y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

#학습을 위한 설정값 전역변수
learning_rate = 0.001
num_epochs = 30
batch_size = 256
display_step = 1

#두개의 히든레이어층으로 만들겠다 
input_size = 784
hidden1_size = 256
hidden2_size = 256
output_size = 10

#tf.data 를 이용해 섞고 배치형태로 가져온다
train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train = train.shuffle(60000).batch(batch_size) #한번 에포가 돌때마다 셔플해준다

def random_normal_initializer_with_stddev_1(): #초기 바이어스 값 선정
    return tf.keras.initializers.RandomNormal(mean=0.0,stddev=1.0,seed=None)

#tf.keras.Model을 이용해 ann 구현
class ANN(tf.keras.Model):
    def __init__(self):
        super(ANN, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(hidden1_size,
        activation='relu',
        kernel_initializer = random_normal_initializer_with_stddev_1(),
        bias_initializer = random_normal_initializer_with_stddev_1())

        self.hidden_layer_2 = tf.keras.layers.Dense(hidden2_size,
        activation = 'relu',
        kernel_initializer = random_normal_initializer_with_stddev_1(),
        bias_initializer = random_normal_initializer_with_stddev_1())

        self.output_layer = tf.keras.layers.Dense(output_size,
        activation=None,)

    def call(self, x):
        H1_output = self.hidden_layer_1(x)
        H2_output = self.hidden_layer_2(H1_output)
        logits = self.output_layer(H2_output)
        return logits
#솔실함수
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.optimizers.Adam(learning_rate)

#최적화 펑션
@tf.function
def train_step(model, x,y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = cross_entropy_loss(y_pred, y)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

#정확도 출력
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

#model declare
ANN_model = ANN()
for epoch in range(num_epochs):
    average_loss = 0.
    total_batch = int(x_train.shape[0]/batch_size)
    for batch_x, batch_y in train:
        _, current_loss = train_step(ANN_model, batch_x,batch_y), cross_entropy_loss(ANN_model(batch_x),batch_y)
        average_loss += current_loss/total_batch
        if epoch % display_step ==4:
            print('epoch %d, loss %f' % ((epoch+1),average_loss))

print(compute_accuracy(ANN_model(x_test), y_test).numpy())
# print('acc %f'%compute_accuracy(ANN_model(x_test), y_test))