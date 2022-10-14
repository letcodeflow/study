import tensorflow as tf
tf.compat.v1.set_random_seed(3289)

x_train_data= [1,2,3]
y_train_data= [3,5,7]

x_train = tf.placeholder(tf.float32,shape=[None])
y_train = tf.placeholder(tf.float32,shape=[None])
#웨이트값 난수로
W = tf.Variable(tf.random_normal([1]),tf.float32)
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
model = x_train*W+b

loss = tf.reduce_mean(tf.square(model-y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.176)

train = optimizer.minimize(loss)
loss_val_list = []
W_val_list = []

epochs = 101
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(epochs):
        _,loss_val,W_val,b_val = sess.run([train, loss,W,b],feed_dict={x_train:x_train_data,y_train :y_train_data})
        if step % 10 == 0:
            print(step,loss_val,W_val,b_val)
        loss_val_list.append(loss_val)
        W_val_list.append(W_val)
######################predict
    x_test_data = [6,7,8]
    x_test = tf.placeholder(tf.float32,shape=[None])

    pred = x_test * W_val + b_val
    # sess = tf.Session()
    # sess.run(pre,feed_dict=x_data,y_test)

    # pred = model.predict(x_test)
    print('[6,7,8] 예측', sess.run(pred,feed_dict={x_test:x_test_data}))


import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# plt.suptitle("Loss & Weight", fontsize=20)
plt.subplot(2,1,1)
plt.plot(W_val_list)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.subplot(2,1,2)
plt.title('ksjjslk')
plt.plot(loss_val_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


# plt.figure(figsize=(10, 10))
# plt.suptitle("Loss & Weight", fontsize=20)
# plt.subplot(2,1,1)
# plt.plot(loss_val_list)
# plt.xlabel("epochs")
# plt.ylabel("loss")


# plt.subplot(2,1,2)
# plt.plot(W_val_list, color = "darkgreen")
# plt.xlabel("epochs")
# plt.ylabel("Weights")


# plt.show()

# plt.subplot(2, 1, 1)
# plt.plot(loss_val_list, color='green', linestyle="--")
# plt.title('loss_val_list')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.subplot(2, 1, 2)
# plt.plot(W_val_list, color='red', linestyle="--")
# plt.title('W_val_list')
# plt.xlabel('epochs')
# plt.ylabel('Weights')
# plt.show()