#1. 데이터 불러오기

import pandas as pd
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submission.csv')




train_set['Street'] = train_set['Street'].astype('category').cat.codes
train_set['MSZoning'] = train_set['MSZoning'].astype('category').cat.codes
train_set['Alley'] = train_set['Alley'].astype('category').cat.codes
train_set['LotShape'] = train_set['LotShape'].astype('category').cat.codes
train_set['LandContour'] = train_set['LandContour'].astype('category').cat.codes
train_set['Utilities'] = train_set['Utilities'].astype('category').cat.codes
train_set['LotConfig'] = train_set['LotConfig'].astype('category').cat.codes
train_set['LandSlope'] = train_set['LandSlope'].astype('category').cat.codes
train_set['Neighborhood'] = train_set['Neighborhood'].astype('category').cat.codes
train_set['Condition1'] = train_set['Condition1'].astype('category').cat.codes
train_set['Condition2'] = train_set['Condition2'].astype('category').cat.codes
train_set['BldgType'] = train_set['BldgType'].astype('category').cat.codes
train_set['HouseStyle'] = train_set['HouseStyle'].astype('category').cat.codes
train_set['RoofStyle'] = train_set['RoofStyle'].astype('category').cat.codes
train_set['RoofMatl'] = train_set['RoofMatl'].astype('category').cat.codes
train_set['Exterior1st'] = train_set['Exterior1st'].astype('category').cat.codes
train_set['Exterior2nd'] = train_set['Exterior2nd'].astype('category').cat.codes
train_set['MasVnrType'] = train_set['MasVnrType'].astype('category').cat.codes
train_set['ExterQual'] = train_set['ExterQual'].astype('category').cat.codes
train_set['ExterCond'] = train_set['ExterCond'].astype('category').cat.codes
train_set['Foundation'] = train_set['Foundation'].astype('category').cat.codes
train_set['BsmtQual'] = train_set['BsmtQual'].astype('category').cat.codes
train_set['BsmtCond'] = train_set['BsmtCond'].astype('category').cat.codes
train_set['BsmtExposure'] = train_set['BsmtExposure'].astype('category').cat.codes
train_set['BsmtFinType1'] = train_set['BsmtFinType1'].astype('category').cat.codes
train_set['BsmtFinType2'] = train_set['BsmtFinType2'].astype('category').cat.codes
train_set['Heating'] = train_set['Heating'].astype('category').cat.codes
train_set['HeatingQC'] = train_set['HeatingQC'].astype('category').cat.codes
train_set['CentralAir'] = train_set['CentralAir'].astype('category').cat.codes
train_set['Electrical'] = train_set['Electrical'].astype('category').cat.codes
train_set['KitchenQual'] = train_set['KitchenQual'].astype('category').cat.codes
train_set['Functional'] = train_set['Functional'].astype('category').cat.codes
train_set['FireplaceQu'] = train_set['FireplaceQu'].astype('category').cat.codes
train_set['GarageType'] = train_set['GarageType'].astype('category').cat.codes
train_set['GarageFinish'] = train_set['GarageFinish'].astype('category').cat.codes
train_set['GarageQual'] = train_set['GarageQual'].astype('category').cat.codes
train_set['GarageCond'] = train_set['GarageCond'].astype('category').cat.codes
train_set['PavedDrive'] = train_set['PavedDrive'].astype('category').cat.codes
train_set['PoolQC'] = train_set['PoolQC'].astype('category').cat.codes
train_set['Fence'] = train_set['Fence'].astype('category').cat.codes
train_set['MiscFeature'] = train_set['MiscFeature'].astype('category').cat.codes
train_set['SaleType'] = train_set['SaleType'].astype('category').cat.codes
train_set['SaleCondition'] = train_set['SaleCondition'].astype('category').cat.codes
train_set['LotFrontage'] = train_set['LotFrontage'].astype('category').cat.codes
train_set['MasVnrArea'] = train_set['MasVnrArea'].astype('category').cat.codes
train_set['GarageYrBlt'] = train_set['GarageYrBlt'].astype('category').cat.codes


print(train_set.info())



#na값 제거하려했으나 의미있는 데이터로 판단
#  6   Alley          91 non-null     object
#   72  PoolQC         7 non-null      object
#  73  Fence          281 non-null    object
#  74  MiscFeature    54 non-null     object
#   57  FireplaceQu    770 non-null    object
  
# # train_set = train_set.drop(['Alley', 'poolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1) 드랍은 바로적용이 안됨



print(train_set.shape)
print(test_set.shape)
test_set['Street'] = test_set['Street'].astype('category').cat.codes
test_set['MSZoning'] = test_set['MSZoning'].astype('category').cat.codes
test_set['Alley'] = test_set['Alley'].astype('category').cat.codes
test_set['LotShape'] = test_set['LotShape'].astype('category').cat.codes
test_set['LandContour'] = test_set['LandContour'].astype('category').cat.codes
test_set['Utilities'] = test_set['Utilities'].astype('category').cat.codes
test_set['LotConfig'] = test_set['LotConfig'].astype('category').cat.codes
test_set['LandSlope'] = test_set['LandSlope'].astype('category').cat.codes
test_set['Neighborhood'] = test_set['Neighborhood'].astype('category').cat.codes
test_set['Condition1'] = test_set['Condition1'].astype('category').cat.codes
test_set['Condition2'] = test_set['Condition2'].astype('category').cat.codes
test_set['BldgType'] = test_set['BldgType'].astype('category').cat.codes
test_set['HouseStyle'] = test_set['HouseStyle'].astype('category').cat.codes
test_set['RoofStyle'] = test_set['RoofStyle'].astype('category').cat.codes
test_set['RoofMatl'] = test_set['RoofMatl'].astype('category').cat.codes
test_set['Exterior1st'] = test_set['Exterior1st'].astype('category').cat.codes
test_set['Exterior2nd'] = test_set['Exterior2nd'].astype('category').cat.codes
test_set['MasVnrType'] = test_set['MasVnrType'].astype('category').cat.codes
test_set['ExterQual'] = test_set['ExterQual'].astype('category').cat.codes
test_set['ExterCond'] = test_set['ExterCond'].astype('category').cat.codes
test_set['Foundation'] = test_set['Foundation'].astype('category').cat.codes
test_set['BsmtQual'] = test_set['BsmtQual'].astype('category').cat.codes
test_set['BsmtCond'] = test_set['BsmtCond'].astype('category').cat.codes
test_set['BsmtExposure'] = test_set['BsmtExposure'].astype('category').cat.codes
test_set['BsmtFinType1'] = test_set['BsmtFinType1'].astype('category').cat.codes
test_set['BsmtFinType2'] = test_set['BsmtFinType2'].astype('category').cat.codes
test_set['Heating'] = test_set['Heating'].astype('category').cat.codes
test_set['HeatingQC'] = test_set['HeatingQC'].astype('category').cat.codes
test_set['CentralAir'] = test_set['CentralAir'].astype('category').cat.codes
test_set['Electrical'] = test_set['Electrical'].astype('category').cat.codes
test_set['KitchenQual'] = test_set['KitchenQual'].astype('category').cat.codes
test_set['Functional'] = test_set['Functional'].astype('category').cat.codes
test_set['FireplaceQu'] = test_set['FireplaceQu'].astype('category').cat.codes
test_set['GarageType'] = test_set['GarageType'].astype('category').cat.codes
test_set['GarageFinish'] = test_set['GarageFinish'].astype('category').cat.codes
test_set['GarageQual'] = test_set['GarageQual'].astype('category').cat.codes
test_set['GarageCond'] = test_set['GarageCond'].astype('category').cat.codes
test_set['PavedDrive'] = test_set['PavedDrive'].astype('category').cat.codes
test_set['PoolQC'] = test_set['PoolQC'].astype('category').cat.codes
test_set['Fence'] = test_set['Fence'].astype('category').cat.codes
test_set['MiscFeature'] = test_set['MiscFeature'].astype('category').cat.codes
test_set['SaleType'] = test_set['SaleType'].astype('category').cat.codes
test_set['SaleCondition'] = test_set['SaleCondition'].astype('category').cat.codes
test_set['LotFrontage'] = test_set['LotFrontage'].astype('category').cat.codes
test_set['MasVnrArea'] = test_set['MasVnrArea'].astype('category').cat.codes
test_set['GarageYrBlt'] = test_set['GarageYrBlt'].astype('category').cat.codes







#1컬럼 차이  saleprice로 예상
print(test_set.info())

# #결측치 삭제하면서 트레인셋에서 x,y분리
# x = train_set.drop(['SalePrice'], axis=1)
# y = train_set['SalePrice']
# print(x.shape)
# print(y.shape)
# #널값 제거


x = train_set.drop(['SalePrice'], axis=1)
y = train_set['SalePrice']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle = False)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.max(x_train))
print(np.min(x_test))
print(np.max(x_test))
#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=80))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(200))
model.add(Dense(1))
import time
#3.컴파일 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
model.fit(x_train, y_train, epochs = 22, batch_size=200, verbose=1)
end_time = time.time()

print('시간', end_time - start_time)
#4.평가 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2', r2)
print(y_predict)
#진짜 답안지
y_summit = model.predict(test_set)
submission['SalePrice'] = y_summit
submission = submission.fillna(0)
submission.to_csv(path + 'submission.csv', index=False)

# loss 735502976.0
# 5/5 [==============================] - 0s 3ms/step
# r2 0.8595986022948693
# 46/46 [==============================] - 0s 731us/step

# loss 661964096.0
# 5/5 [==============================] - 0s 3ms/step
# r2 0.8736365850993209
# 46/46 [==============================] - 0s 537us/step

#민맥스
# loss 2358610432.0
# r2 0.549760912272991

# 스탠
# loss 950847104.0
# r2 0.8184911978215939

