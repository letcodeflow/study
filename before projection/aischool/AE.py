#keras subclassing
import tensorflow as tf
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#타입변경
x_train, x_test = x_train.astype('float32'),x_test.astype('float32')
#flattening
x_train, x_test = x_train.reshape([-1,784]),x_test.reshape([-1,784])
#normalize
x_train, x_test = x_train/255., x_test/255.

#학습을 위한 설정값 전역변수
learning_rate = 0.02
training_epochs = 50
num_epochs = 1
batch_size = 256
display_step = 1
example_to_show = 10
#두개의 히든레이어층으로 만들겠다 
input_size = 784
hidden1_size = 128
hidden2_size = 256
output_size = 10

train = tf.data.Dataset.from_tensor_slices(x_train) #x데이터만 갖고옴
train = train.shuffle(60000).batch(batch_size)

def random_normal_initializer_with_stddev_1():
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        #784 - 256-128
        self.hidden_layer_1 = tf.keras.layers.Dense(hidden1_size,
        activation='sigmoid',
        kernel_initializer = random_normal_initializer_with_stddev_1(),
        bias_initializer = random_normal_initializer_with_stddev_1())

        self.hidden_layer_2 = tf.keras.layers.Dense(hidden2_size,
        activation='sigmoid',
        kernel_initializer = random_normal_initializer_with_stddev_1(),
        bias_initializer = random_normal_initializer_with_stddev_1())

        #decoding 128 - 256 784
        self.hidden_layer_3 = tf.keras.layers.Dense(hidden1_size,
        activation = 'sigmoid',
        kernel_initializer = random_normal_initializer_with_stddev_1(),
        bias_initializer = random_normal_initializer_with_stddev_1())

        self.output_layer = tf.keras.layers.Dense(input_size,
        activation = 'sigmoid',
        kernel_initializer = random_normal_initializer_with_stddev_1(),
        bias_initializer = random_normal_initializer_with_stddev_1())

    def call(self, x):
        H1_output = self.hidden_layer_1(x)
        H2_output = self.hidden_layer_2(H1_output)
        H3_output = self.hidden_layer_3(H2_output)
        reconstructed_x = self.output_layer(H3_output)

        return reconstructed_x

@tf.function
def mse_loss(y_pred, y_true):
    return tf.reduce_mean(tf.pow(y_true-y_pred, 2)) #차이를 제곱해서 평균

optimizer = tf.optimizers.RMSprop(learning_rate)

@tf.function
def train_step(model, x):
    y_true = x
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = mse_loss(y_pred, y_true)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

AE_model = AutoEncoder()

for epoch in range(training_epochs):
    for batch_x in train: #비지도 y 필요없음
        _, current_loss = train_step(AE_model, batch_x), mse_loss(AE_model(batch_x),batch_x) #파라미터 업데이트
        if epoch % display_step == 0:
            print('epoch %d, loss %f' %((epoch+1), current_loss))

#테스트데이터로 reconstruction
reconstructed_result = AE_model(x_test[:example_to_show])
import numpy as np
#비교
f, a =  plt.subplots(2,10,figsize=(10,2))
for i in range(example_to_show):
    a[0][i].imshow(np.reshape(x_test[i],(28,28)))
    a[1][i].imshow(np.reshape(reconstructed_result[i],(28,28)))
f.savefig('reconstructed_mnist_image.png')
f.show()
plt.axis('off ')
plt.show()
plt.waitforbuttonpress()