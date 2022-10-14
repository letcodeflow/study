import tensorflow as tf
tf.random.set_seed(13)
import pandas as pd
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'gender_submission.csv', index_col=0)

print(train_set.columns)
print(train_set)
print(train_set.head())
print(list(train_set.columns))
print(train_set)
print(test_set)
print(submission)
columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
for column in columns:
    print(column)
    print(train_set[column].unique())
    print(test_set[column].unique())

train_set = pd.get_dummies(train_set, columns=['Sex', 'Embarked'])
test_set = pd.get_dummies(test_set, columns=['Sex', 'Embarked'])

print(test_set.isnull().sum())

train_set = train_set.drop(['Cabin', 'Name', 'Age', 'Ticket'], axis=1)
test_set = test_set.drop(['Cabin', 'Name', 'Age', 'Ticket'], axis=1)

x = train_set.drop(['Survived'], axis=1)
y = train_set['Survived']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=137)
#트레인셋을 나누기 전에 쓰면 과적합의 위허미 있기때문
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
import numpy as np
print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))


from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

input1 = Input(shape=(9,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(10, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
dense5 = Dense(10, activation='relu')(dense4)
output1 = Dense(1, activation='sigmoid')(dense5)
model = Model(inputs=input1, outputs=output1)
model.summary()

#3.컴파일 훈련
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filename='{epoch:04d}_{val_loss:.4f}.hdf5'
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
ES = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
MCP = ModelCheckpoint(monitor='val_loss', verbose=1, save_best_only=True, mode='auto', filepath='./_data/kaggle_titanic/k25/'.join([date,'_',filename]))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, batch_size=20, epochs=10, callbacks=[ES,MCP], validation_split=0.2)

#4.평가 예측
loss = model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
y_summit = model.predict(test_set)
#리니어값
y_predict = y_predict.flatten()                 
y_predict = np.where(y_predict >= 0.5, 1 , 0) 
y_summit = np.maximum(0, y_summit)
y_summit = y_summit.round()
print(y_predict)
print(y_summit)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)

submission['Survived'] = y_summit.astype(int, copy=True)
submission.to_csv(path + 'submission.csv')


