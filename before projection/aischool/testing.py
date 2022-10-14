import tensorflow as tf

W = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))

#가설정의
@tf.function
def linear_model(x):
    return W*x +b

#손실함수 정의
#mse
@tf.function
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred-y))

#옵티마이저 정의
optimizer = tf.optimizers.SGD(0.01)

#최적화 기능
@tf.function
def train_step(x,y): #1epoch = train_step
    with tf.GradientTape() as tape: #scope
        y_pred = linear_model(x)
        loss = mse_loss(y_pred, y)
        gradients = tape.gradient(loss, [W,b]) 
        optimizer.apply_gradients(zip(gradients, [W,b]))


x_train = [1,2,3,4]
y_train = [2,4,6,8]

for i in range(1000):
    train_step(x_train,y_train)

#test
x_test = [3.5,5,5.5,6]
print(linear_model(x_test))


