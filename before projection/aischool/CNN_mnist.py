#keras subclassing
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


train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train = train.repeat().shuffle(60000).batch(50)
train = iter(train)

class CNN(tf.keras.Model):
    def __init__(self):
       super(CNN, self).__init__()
       self.conv_layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5,strides=1, padding='same',activation='relu')
       self.pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2)

       self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5,strides=1,padding='same',activation='relu')
       self.pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2)

       #fully connected
       #7x7 width height의 64개 activation map을 1024 개의 feature로 변환
       self.flatten_layer = tf.keras.layers.Flatten()
       self.fc_layer_1 = tf.keras.layers.Dense(1024, activation='relu')

       #output
       #1024 features 를 10개 클래스 one hot
       self.output_layer = tf.keras.layers.Dense(10, activation=None)

    def call(self, x):
        #3차원으로ㅓ reshape, grayscale이기 때문에 채널1
        x_image = tf.reshape(x,[-1,28,28,1])
        #28 x 28x 1 -> 28x28x32
        h_conv1 = self.conv_layer_1(x_image)
        #2828x32 -> 14x14x32
        h_pool1 = self.pool_layer_1(h_conv1)
        # 14x14x32 -> 7x7x64
        h_conv2 = self.conv_layer_2(h_pool1)
        #7x7x64(3136) ->1024
        h_pool2 = self.pool_layer_2(h_conv2)
        h_pool2_flat = self.flatten_layer(h_pool2)
        h_fc1 = self.fc_layer_1(h_pool2_flat)
        #1024-10
        logits = self.output_layer(h_fc1)
        y_pred = tf.nn.softmax(logits)

        return y_pred, logits

#cross entropy
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.optimizers.Adam(1e-4) #scientific notation

@tf.function
def train_step(model, x,y):
    with tf.GradientTape() as tape:
        y_pred, logits = model(x)
        loss = cross_entropy_loss(logits, y)
        gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))

#acc
@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

CNN_model = CNN()  

for i in range(10000):
    batch_x, batch_y = next(train)
    if i %100 ==0:
        train_acc = compute_accuracy(CNN_model(batch_x)[0], batch_y)
        print('epoch %d, acc %f'%(i, train_acc))
    train_step(CNN_model,batch_x,batch_y)

print('acc %f' %compute_accuracy(CNN_model(x_test)[0],y_test))