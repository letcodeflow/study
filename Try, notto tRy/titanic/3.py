import matplotlib


웨이트난수값 지정
데이터불러오기 3개다 가져올때 index=0할것이냐
columns
value counts
np unique return
shape
head
columns.values
print(list(train_set.columns))
tocategorical로 sex embarked 변환해서 칼럼지정

결측치
object 확인
describe, info , innulld
embarked s채움
테스트셋도 적용
테스트셋에 빈값있으면 dropna로는 행없어짐 fillnatrain_set['Embarked'] = train_set['Embarked'].fillna(train_set.Embarked.dropna().mode()[0])    # Embarked 부분을 지우고 
value=train_set.Age.mean()
train_set['Embarked'] = train_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)           # 'S':0, 'C':1, 'Q':2 로 매핑한다 int로

y_predict = model.predict(x_test)
print(y_predict)
y_predict = y_predict.flatten()                 
y_predict = np.where(y_predict > 0.5, 1 , 0)   
print(y_predict) 

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict)) 

y_summit = model.predict(test_set)
y_summit = y_summit.flatten()                 
y_summit = np.where(y_summit > 0.5, 1 , 0)   


# y_test = np.argmax(y_test,axis=1)
# import tensorflow as tf
# y_test = np.argmax(y_test,axis=1)
# y_predict = np.argmax(y_predict,axis=1)

트레인 테스트셋
스케일러 트레인셋 이후 쓰는 이유
함수모델 액티베이션
모델서머리
가중치파일 저장

얼리스타핑 메트릭스
평가 x_test넣은 y_predict test_set 자체를 넣은 y_summit
print
0.5>=
y_predict = y_predict.round(0)
빈값 채워줘야 하나?
submission tocsv
.astype(int)
index=
matplotlib 한글 제목
font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


