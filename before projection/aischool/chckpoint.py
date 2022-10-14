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

#param save
SAVER_DIR = './model'
ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=CNN_model) #처음반복회수 지정 
ckpt_manager = tf.train.CheckpointManager( #keep 최근 몇개까지 저장할거냐
    ckpt,directory=SAVER_DIR, max_to_keep=5
)

latest_ckpt = tf.train.latest_checkpoint(SAVER_DIR) #최근 ckpt full path 리턴
print(latest_ckpt)
if latest_ckpt: #파일있으면 True
    ckpt.restore(latest_ckpt) #복원
    print('테스트 정확도 restore %f' %compute_accuracy(CNN_model(x_test)[0], y_test)) #불러온 값으로 테스트정확도 측정 종료
    exit() #삭제하면 이어서 학습

while int(ckpt.step) <(10000+1): #전역 tf.variable 로 가져온뒤 tensor 형태이므로 int값  횟수
    batch_x, batch_y = next(train)
    #100step마다 트레인셋에 대한 정확도 출력하고 매니저를 이용해 파람 저장
    if ckpt.step % 100==0:
        ckpt_manager.save(checkpoint_number=ckpt.step) #실제저장
        train_acc = compute_accuracy(CNN_model(batch_x)[0],batch_y)
        print('epoch %d acc %f'%(ckpt.step, train_acc))
    train_step(CNN_model,batch_x,batch_y)
    
    ckpt.step.assign_add(1) #epoch끝날때마다 

# for i in range(10000):
#     batch_x, batch_y = next(train)
#     if i %100 ==0:
#         train_acc = compute_accuracy(CNN_model(batch_x)[0], batch_y)
#         print('epoch %d, acc %f'%(i, train_acc))
#     train_step(CNN_model,batch_x,batch_y)

print('acc %f' %compute_accuracy(CNN_model(x_test)[0],y_test))