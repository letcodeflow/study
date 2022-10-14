#피처 하나 삭제하고 성능비교
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape)
x = x[:,[0,2,3,4,5,6,7,8,9]]
from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,random_state=234)


#2. 모델구성
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = XGBRegressor()

model.fit(x_train, y_train)

#4평가예측
from sklearn.metrics import r2_score
y_predicy = model.predict(x_test)
r2 = r2_score(y_test, y_predicy)
result = model.score(x_test,y_test)
print('model.score',result)
print('r2',r2)
print(model.feature_importances_)

# Decision model.score -0.1881727749447688
# r2 -0.1881727749447688
# [0.05245723 0.00628897 0.25955154 0.11003986 0.01140431 0.0638642
#  0.05688743 0.02734165 0.33649277 0.07567204]

# RandomForest model.score 0.42026809959984135
# r2 0.42026809959984135
# [0.06179878 0.01291196 0.29592849 0.10820925 0.04121135 0.06106726
#  0.05203309 0.02586473 0.28374708 0.05722801]

# gradientbosting model.score 0.3949138285469649
# r2 0.3949138285469649
# [0.05532071 0.02437011 0.23036701 0.10745296 0.0310961  0.06290015
#  0.03353306 0.01595972 0.40484397 0.03415621]

# xgb model.score 0.2671757230781665
# r2 0.2671757230781665
# [0.0292707  0.07425479 0.16275044 0.08692391 0.03762532 0.05480872
#  0.05896411 0.01974154 0.42998356 0.04567682]


# 1컬럼 삭제

# xgb model.score 0.17721203344958214
# r2 0.17721203344958214
# [0.03188016 0.13994919 0.07927788 0.04208064 0.06314535 0.06239991
#  0.04998838 0.47231683 0.05896159]

# gradient model.score 0.39772870503525315
# r2 0.39772870503525315
# [0.05318119 0.23943098 0.09447561 0.04037958 0.07753037 0.02503081
#  0.01645235 0.4113461  0.04217302]

# radom model.score 0.39640425332687146
# r2 0.39640425332687146
# [0.06267539 0.29302983 0.11393266 0.04306659 0.06185412 0.0554495
#  0.02285577 0.2841571  0.06297904]

# decsion model.score -0.10751555135403179
# r2 -0.10751555135403179
# [0.04495137 0.24236366 0.12145996 0.02428999 0.0820696  0.0439195
#  0.0275191  0.34025164 0.07317518]