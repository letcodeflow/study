import tensorflow as tf
#초기 랜덤웨이트값 정의

from subprocess import call
import pandas as pd

#describe info insullsum 
#y라벨의 종류가 무엇인지 확인하는 판다스 함수 =  np.unique
path = 'c:/study/_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

print(train_set.shape)
print(test_set.shape)
print(test_set.columns)
print(train_set.columns)
print(test_set.columns.values)
print(train_set.columns.values)
#values 컬럼명의 데이터타입이 생략됨


# pd.Series.unique(train_set)
# pd.Series.value_counts(train_set)
# (891, 12)
# (418, 11)
# Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
#        'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')
# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')
# print(train_set.info())
# print(test_set.info())
# print(train_set.describe())
# print(test_set.describe())
 #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   PassengerId  891 non-null    int64
#  1   Survived     891 non-null    int64
#  2   Pclass       891 non-null    int64
#  3   Name         891 non-null    object
#  4   Sex          891 non-null    object
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
#  8   Ticket       891 non-null    object
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object
#  11  Embarked     889 non-null    object
 
 
#  ---  ------       --------------  -----
#  0   PassengerId  418 non-null    int64
#  1   Pclass       418 non-null    int64
#  2   Name         418 non-null    object
#  3   Sex          418 non-null    object
#  4   Age          332 non-null    float64
#  5   SibSp        418 non-null    int64
#  6   Parch        418 non-null    int64
#  7   Ticket       418 non-null    object
#  8   Fare         417 non-null    float64
#  9   Cabin        91 non-null     object
#  10  Embarked     418 non-null    object

print(train_set.isnull().sum())
print(test_set.isnull().sum())
print(train_set['Survived'].unique())
#서바이브드 고유라벨값이 무엇인지
print(train_set['Survived'].value_counts())
#value counts쓰면 한번에 나오는듯? 라벨값이 뭐고 몇개인지
print(train_set['Embarked'].value_counts())


train_set = pd.get_dummies(train_set, columns = ['Sex', 'Embarked'])
print(list(train_set.columns))
#트레인셋

print(train_set.head())
print(train_set.columns.values)


train_set=train_set.dropna()
test_set=test_set.dropna()



#일단 떨군다


#        PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
# count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
# mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
# std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
# min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
# 25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
# 50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
# 75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
# max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
#        PassengerId      Pclass         Age       SibSp       Parch        Fare
# count   418.000000  418.000000  332.000000  418.000000  418.000000  417.000000
# mean   1100.500000    2.265550   30.272590    0.447368    0.392344   35.627188
# std     120.810458    0.841838   14.181209    0.896760    0.981429   55.907576
# min     892.000000    1.000000    0.170000    0.000000    0.000000    0.000000
# 25%     996.250000    1.000000   21.000000    0.000000    0.000000    7.895800
# 50%    1100.500000    3.000000   27.000000    0.000000    0.000000   14.454200
# 75%    1204.750000    3.000000   39.000000    1.000000    0.000000   31.500000
# max    1309.000000    3.000000   76.000000    8.000000    9.000000  512.329200


#패신저 아이디 삭제

#xy 설정하고 테스트 트레인셋 나눈고 모델구성 


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

x = train_set.drop(['Survived','Name', 'Ticket', 'Cabin'], axis=1)
y = train_set['Survived'].values.reshape(-1,1)
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=137)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
# scaler = MinMaxScaler()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
model_input_var1 = RandomForestRegressor(criterion = 'mse')
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))
print(x_train.shape, x_test.shape)
model = Sequential()
import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score
x = tf.compat.v1.placeholder(tf.float32,shape=[None,10])
y = tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random.normal([10,18]),tf.float32)
b = tf.Variable(tf.random.normal([18]),tf.float32)
hidden = tf.matmul(x,w)+b

w2 = tf.Variable(tf.random.normal([18,32]),tf.float32)
b2 = tf.Variable(tf.random.normal([32]),tf.float32)
hidden2 = tf.matmul(hidden,w2)+b2

w3 = tf.Variable(tf.random.normal([32,16]),tf.float32)
b3 = tf.Variable(tf.random.normal([16]),tf.float32)
hidden3 = tf.matmul(hidden2,w3)+b3

w4 = tf.Variable(tf.random.normal([16,1]),tf.float32)
b4 = tf.Variable(tf.random.normal([1]),tf.float32)
hypothesis = tf.sigmoid(tf.matmul(hidden3,w4)+b4)
loss = tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.00000000000000000000001e-15).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        _ = sess.run(train,feed_dict={x:x_train,y:y_train})
        if i % 2==0:
            print(i)
    pred = sess.run(hypothesis,feed_dict = {x:x_test})
    # pred = np.argmax(pred,axis=1)
    # y_test = np.argmax(y_test,axis=1)
    print(accuracy_score(y_test,pred))

#현재 문제 y_predict값 nan값 나옴 이해못함 아마도 라벨인코딩 argmax 조합인것 같은데 다시 봐야함
#pandas 함수 적용못함

# y_summit = model.predict(test_set)
# submission = pd.read_csv(path + 'gender_submission.csv')
# submission['Survive'] = y_summit
# submission.to_csv(path + 'submission.csv')
# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).

#현재 라벨인코딩 밋 원핫인코딩 3개 쓸수 없음으로 인해서 데이터셋 na를 전부 날렸다 알게되면 원핫인코딩으로 숫자변환해 쓸수 있을 것 추후 전처리를 배우면 상관관계를 분석해 이용할 수?
#argmax 의 의의를 제대로 이해못함
#아침에 와서 모든 기술이 다 들어간 코드 작성해보기

# 미적용
# loss [0.5197330117225647, 0.7857142686843872]
# acc 0.26785714285714285

# 민맥스
# loss [0.45250099897384644, 0.7321428656578064]
# acc 0.26785714285714285

# 스탠
# loss [0.4650607705116272, 0.7678571343421936]
# acc 0.26785714285714285

# dropout
# loss [0.45394083857536316, 0.7857142686843872]
# acc 0.26785714285714285