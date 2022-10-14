#keras subclassing
from numpy import gradient
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#타입변경
x_train, x_test = x_train.astype('float32'),x_test.astype('float32')
#flattening
x_train, x_test = x_train.reshape([-1,784]),x_test.reshape([-1,784])
#normalize
x_train, x_test = x_train/255., x_test/255.
#one-hot
y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

#tf.data를 이용해 데이터를 섞고 batch 형태로 가져온다
train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train = train.repeat().shuffle(60000).batch(100)
train_iter = iter(train)

# print(list(train_iter)) 

#tf.keras.Model 을 통해 Softmax reg 정의
class SoftmaxRegression(tf.keras.Model):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        self.softmax_layer = tf.keras.layers.Dense(10,activation=None, #784->10
        kernel_initializer = 'zeros',
        bias_initializer='zeros')

    def call(self, x): #자동실행 인자값으로 인풋을 받은후 softmax 처리 출력
        logits = self.softmax_layer(x)
        return tf.nn.softmax(logits)

#cross entropy 정의
@tf.function
def cross_entropy_loss(y_pred, y):
    return tf.reduce_mean(-tf.reduce_sum(y* tf.math.log(y_pred),[1] ))
    #return tf.reduce_mean(tf.nn.softmax_cross_entropy_logits(logits=logits, label=y)) 
    #쓸거면 logit을 그대로 가져와야함

optimizer = tf.optimizers.SGD(0.5)

#최적화
@tf.function
def train_step(model,x,y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = cross_entropy_loss(y_pred,y)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 

#정확도
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

#softmax reg model declare
SoftmaxRegression_model = SoftmaxRegression()
for i in range(1000):
    batch_xs, batch_ys = next(train_iter)
    train_step(SoftmaxRegression_model, batch_xs,batch_ys)

print('acc %f'%compute_accuracy(SoftmaxRegression_model(x_test), y_test))