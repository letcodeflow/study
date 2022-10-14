import tensorflow as tf
tf.set_random_seed(1324)

t = tf.Variable(tf.random_normal([1]),name='wljnb')
print(t)

#1.초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(t)
print(aaa)
sess.close()

#2. 두번째 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())
bbb = t.eval(session=sess)
print(bbb)
sess.close()

#3. 세번재
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
ccc = t.eval()
print(ccc)
sess.close()