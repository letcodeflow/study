from unittest import TestCase
import tensorflow as tf
from torch import gradient

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#data type change
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
#0,1 사이 값으로 Normalize
x_train, x_test = x_train/255., x_test/255.
#스칼라 형태 레이블 0~9를 one hot으로 변환
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),1) #squeeze 더미 디멘션 삭제 <_> tf.expand_dims 1장씩 테스트할때 주로씀
y_test_one_hot = tf.squeeze(tf.one_hot(y_test,10),1)

#tf.data 로 섞고 배치로 가져온다
train = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
train = train.repeat().shuffle(50000).batch(128)
train_iter = iter(train)

test = tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
test = test.batch(1000) #메모리용량대문에 배치설정
test_iter = iter(test)

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool_layer1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        
        self.conv_layer2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool_layer2 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2)

        self.conv_layer3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')

        self.conv_layer4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')

        self.conv_layer5 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
        #8x8x128
        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc_layer1 = tf.keras.layers.Dense(384, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)

        #10class
        self.output_layer = tf.keras.layers.Dense(10,activation=None)

    def call(self, x, is_training): #train 셋인지 bool 값 입력함으로 드롭아웃 적용결정
        h_conv1 = self.conv_layer1(x)
        h_pool1 = self.pool_layer1(h_conv1)
        h_conv2 = self.conv_layer2(h_pool1)
        h_pool2 = self.pool_layer2(h_conv2)
        h_conv3 = self.conv_layer3(h_pool2)
        h_conv4 = self.conv_layer4(h_conv3)
        h_conv5 = self.conv_layer5(h_conv4)
        h_conv5_flat = self.flatten_layer(h_conv5)
        h_fc1 = self.fc_layer1(h_conv5_flat)
        h_fc1_drop = self.dropout(h_fc1, training= is_training)
        #dropout, feature 의 co-adaptation 방지
        logits = self.output_layer(h_fc1_drop)
        y_pred = tf.nn.softmax(logits)

        return y_pred, logits


#cross_entropy
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.optimizers.RMSprop(1e-3)

#최적화펑션
@tf.function
def train_step(model, x,y, is_training):
    with tf.GradientTape() as tape:
        y_pred, logits = model(x, is_training)
        loss = cross_entropy_loss(logits, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@tf.function
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

CNN_model = CNN()

for i in range(10000):
    batch_x, batch_y = next(train_iter)

    if i % 100 ==0:
        train_acc = compute_accuracy(CNN_model(batch_x, False)[0], batch_y)
        loss_print = cross_entropy_loss(CNN_model(batch_x, False)[1], batch_y)
        #dropouyt 적용하지 않고 로스값 뽑기
        print('epoch %d, acc %f, loss %f' %(i, train_acc, loss_print))

    train_step(CNN_model, batch_x,batch_y,True)
        #실제 트레닝시 적용

test_acc = 0.0
for i in range(10):
    test_batch_x, test_batch_y = next(test_iter)
    test_acc = test_acc +compute_accuracy(CNN_model(test_batch_x, False)[0], test_batch_y).numpy()
test_acc = test_acc /10
print('test acc ', test_acc )