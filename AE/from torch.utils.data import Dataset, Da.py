from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import random
import glob
import os

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import warnings
warnings.filterwarnings(action='ignore') 


# --------------------..---------------------------------------------------------------------------------------------------------------------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # GPU or CPU 환경 확인 - 현재 GPU환경이면 cuda, cpu환경이면 cpu출력
print(device) #- cpu

# --------------.---------------------------------------------------------------------------------------------------------------------------------------------------------
CFG = { # 하이퍼 파라미터 셋팅
    'EPOCHS' : 5,        
    'LEARNING_RATE': 1e-3,  
    'BATCH_SIZE' : 16,
    'SEED' : 41
}

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)                         # torchvision의 transform에서 RandomCrop, RandomHorizontalFlip등은 python의 random을 사용하여 시드 고정
    os.environ['PYTHONHASHSEED'] = str(seed)  # 환경변수 관련 : 환경 변수가 0 으로 설정된 경우에만 효과 / 모르겠습니다
    np.random.seed(seed)                      # 파이토치에서 많은 코드가 넘파이로 데이터를 받아오기 때문에 넘파이 시드도 고정
    torch.manual_seed(seed)                   # torch~로 시작하는 모든 코드의 난수 고정
    torch.cuda.manual_seed(seed)              # torch.cuda~로 시작하는 모든 코드의 난수 고정
    torch.backends.cudnn.deterministic = True # 파이토치는 cudnn을 백엔드로 사용하기 때문에 이것도 설정, 속도가 느려질 수 있다
    torch.backends.cudnn.benchmark = True     # True면 convolution 연산을 input size 맞게 최적화된 알고리즘 사용, 단점 입력 이미지 사이즈가 너무 다르면 성능이 저하될 수 있다

seed_everything(CFG['SEED'])                  # seed_everything 함수의 seed를 CFG 파라미터의 41로 선언

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
all_input_list = sorted(glob.glob('./_data/OMG/train_input/*.csv'))   # *로 된 csv파일들을 리스트로 변환(glob.glob)하여 정렬(sorted)시킨다
all_target_list = sorted(glob.glob('._data/OMG/train_target/*.csv'))  # - sorted() : 정렬된 새로운 리스트를 반환
# print(all_input_list)                                               # - sort()   : 리스트 자체를 변경
                                                                      # - *        : 임의 길이의 모든 문자열을 의미
                                                                      # - ?        : 한자리의 문자를 의미
                                                                      # glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
# train_set, validation_set 분리 
train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]
                                                              
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 전처리
class CustomDataset(Dataset) :                                                          # CustomDataset 클래스 정의
    def __init__(self, input_paths, target_paths, infer_mode) :                         # 속성 정의
        self.input_paths = input_paths                                                  # 속성들의 할당값 정의
        self.target_paths = target_paths
        self.infer_mode = infer_mode
        
        self.data_list = []                                                             # data_list 이라는 빈 리스트
        self.label_list = []                                                            # label_list 이라는 빈 리스트
        print('Data Pre-processing..')
        for input_path, target_path in tqdm(zip(self.input_paths, self.target_paths)) : # input_paths와 target_paths를 튜플형태(zip)로 진행상황(tqdm)을 볼 수 있는 for문 정의
            input_df = pd.read_csv(input_path)                                          # 불러들인 판다스로 읽기
            target_df = pd.read_csv(target_path)
            
            input_df = input_df.drop(columns=['시간'])                                  # input_df에서 시간 컬럼 drop
            input_df = input_df.fillna(0)                                               # NaN값을 0으로 
            
            input_length = int(len(input_df)/1440)                                      # input_df의 길이(len)를 1440으로 나눈것을 정수형(int)으로 변환
            target_length = int(len(target_df))                                         # target_df의 길이를 정수형으로 변환
            
            for idx in range(target_length) :                                           # target_length의 길이 만큼 for문 정의
                time_series = input_df[1440*idx:1440*(idx+1)].values                    # input_df을 1440개 단위로 묶어 values값으로 지정
                self.data_list.append(torch.Tensor(time_series))                        # time_series를 텐서로 변환하여 data_list에 더해준다
            for label in target_df["rate"] :                                            # target_df의 rate컬럼의 값들을 label_list에 더해주는 for문 정의
                self.label_list.append(label)  
        print('Done.')
        
              
    def __getitem__(self, index) :    
        data = self.data_list[index]   
        label = self.label_list[index] 
        if self.infer_mode == False :  # infer_mode가 False면
            return data, label         # data와 label return
        else :
            return data                # 아니면 data return
        
    def __len__(self) :
        return len(self.data_list)     # data_list의 길이 return
    
train_dataset = CustomDataset(train_input_list, train_target_list, False)                              # CustomDataset 클래스에 train_input_list, train_target_list 넣는다
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=6) # shuffle=True -> False로 변경[에러]
#                         datasets     , batch_size = CFG의 seed값                    , 서브 프로세서 6개로 사용
val_dataset = CustomDataset(val_input_list, val_target_list, False)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=6) 

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 모델 정의                                                                                          
class BaseModel(nn.Module):                                                                        
    def __init__(self):                                                                            # super().__init__() 다른 클래스의 속성 및 메소드를 자동으로 불러 사용 가능
        super(BaseModel, self).__init__()                                                          # nn 라이브러리 사용하여 LSTM모델 정의 
        self.lstm = nn.LSTM(input_size=37, hidden_size=256, batch_first=True, bidirectional=False) # bidirectional : 양방향 RNN
                                         # hidden_size=은닉층 사이즈, batch_first=True면 output값 사이즈는 (batch, seq, feature) 
        self.classifier = nn.Sequential( # Sequential 정의
            nn.Linear(256, 1),           # Sequential 안에 input 256, output 1인 Linear 계층
        )
        
    def forward(self, x) :                       # 모델이 구현되는 곳
        hidden, _ = self.lstm(x)                 # _... input값 x 는 LSTM을 거친다
        output = self.classifier(hidden[:,-1,:]) # 
        return output
    
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------    
# 훈련
def train(model, optimizer, train_loader, val_loader, scheduler, device) : # 함수 정의
    model.to(device)                                                       # 모델을 GPU로 돌리기 위해 device
    criterion = nn.L1Loss().to(device)                                     # criterion에 L1Loss[절대오차 MAE] 사용
    
    best_loss = 9999                          # best_loss 기준 지정
    best_model = None                         # best_model 아무것도 없는 변수 지정
    for epoch in range(1, CFG['EPOCHS']+1):   # CFG['EPOCHS']=5 range(1, 5)인 for문 정의
        model.train()
        train_loss = []
        for X, Y in tqdm(iter(train_loader)): # 전 처리에서 정의한 반복가능한 train_loader를 iter을 사용하여 진행상황(tqdm)을 볼 수 있는 for문 정의
            X = X.to(device)                  # X를 device
            Y = Y.to(device)
            
            optimizer.zero_grad()             # optimizer 초기화 - zero_grad()를 이용해 옵티마이저에 사용된 파라미터들의 기울기를 0으로 설정
            
            output = model(X)                 # model에 X를 넣어 예측값을 output에 넣는다
            loss = criterion(output, Y)       # 예측값과 실제값의 손실함수를 loss로 측정
            
            loss.backward()                   # loss의 backward()를 호출(역전파 계산 및 grad 속성에 누적)
            optimizer.step()                  # 최적화를 위해 optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정
            
            train_loss.append(loss.item())    # loss를 train_loss에 더해준다
                    
        val_loss = validation(model, val_loader, criterion, device) # validation 함수 리턴값
        
        print(f'Train Loss : [{np.mean(train_loss):.5f}] Valid Loss : [{val_loss:.5f}]') # train_loss 평균값, val_loss 평균값 출력
        
        if scheduler is not None :            # scheduler가 None이 아니라면
            scheduler.step()                  # scheduler... lr 를 점점 감소 시키는 Scheduler 
            
        if best_loss > val_loss :             # val_loss값이 9999보다 작으면 
            best_loss = val_loss              # best_loss = val_loss값
            best_model = model                # best_model = model
    return best_model                         # train 함수 return값은 best_model

def validation(model, val_loader, criterion, device) : 
    model.eval()                             # model을 eval(연산)한다 - 문자열로 된 수식을 Input으로 받아 그 결과를 return 하는 함수
    val_loss = []                                      
    with torch.no_grad() :                   # no_grad()를 사용하여 기울기의 업데이트X
        for X, Y in tqdm(iter(val_loader)) : # 전 처리에 정의한 반복가능한 val_loader를 iter을 사용하여 진행상황(tqdm)을 볼 수 있는 for문 정의
            X = X.float().to(device)         # X를 float로 변환하여 device
            Y = Y.float().to(device)
            
            model_pred = model(X)            # X를 넣어 예측값 반환
            loss = criterion(model_pred, Y)  # 예측값과 실제값 비교하여 loss 반환
            
            val_loss.append(loss.item())     # 실제(item) loss값을 val_loss에 더해준다
            
    return np.mean(val_loss)                 # val_loss의 평균값 return

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------    
# 모델 및 훈련
model = BaseModel()                                                                  # 모델 BaseModel 클래스사용
model.eval()                                                                         # 모델 연산
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"]) # params은 model의 parameters()로 설정m lr은 1e-3
scheduler = None

best_model = train(model, optimizer, train_loader, val_loader, scheduler, device)   

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------  
# 결과
test_input_list = sorted(glob.glob('./test_input/*.csv'))   # *로된 csv파일들을 리스트(glob)로 변환하여 정렬(sorted)한다
test_target_list = sorted(glob.glob('./test_target/*.csv')) 

def inference_per_case(model, test_loader, test_path, device) :
    model.to(device)                 # 모델 device
    model.eval()                     # 모델 연산
    pred_list = []
    with torch.no_grad() :           # no_grad()를 사용하여 기울기의 업데이트X
        for X in iter(test_loader) : # 반복 가능한 test_loader를 iter을 사용하여 꺼내주는 for문 정의
            X = X.float().to(device) # X를 실수타입(float)으로 변환하여 device
            
            model_pred = model(X)    # model에 X를 넣어 예측값 반환
                                                                       # 모르겠습니다
            model_pred = model_pred.cpu().numpy().reshape(-1).tolist() # .cpu() - GPU 메모리에 올려져 있는 tensor를 cpu 메모리로 복사하는 method
                                                                       # .numpy() - tensor를 numpy로 변환하여 반환[이때 저장공간을 공유하기 때문에 하나를 변경하면 다른 하나도 변경된다]
            pred_list += model_pred  # pred_list에 model_pred값을 더해준다
    
    submit_df = pd.read_csv(test_path)       # test_path = test_target_path를 판다스로 읽고
    submit_df['rate'] = pred_list            # pred_list값을 submit_df['rate']에 넣는다
    submit_df.to_csv(test_path, index=False) # submit_df를 test_path에 index제거 후 csv로 변환 
    
for test_input_path, test_target_path in zip(test_input_list, test_target_list) : # test_input_list와 test_target_list를 튜플형태(zip)로 진행상화을 볼 수 있음 for문 정의
    test_dataset = CustomDataset([test_input_path], [test_target_path], True)     # test_input_path와 test_target_path를 list형태로 CustomDataset class에 넣는다
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0) # 서브 프로세서 사용X
                           # datasets    , batch_size = CFG의 seed값
    inference_per_case(best_model, test_loader, test_target_path, device)         # inference_per_case함수에 인자들을 넣는다


# TEST_01.csv
import zipfile
os.chdir("./_data/OMG/")                                   # 경로설정 : Directory 위치 변경
submission = zipfile.ZipFile("sample_submission.zip", 'w') # zip파일 쓰기모드(w)
for path in test_target_list :
    submission.write(path)
submission.close()    

# import zipfile
# filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
# os.chdir("D:\study_data\_data\dacon_vegi/test_target")
# with zipfile.ZipFile("submission.zip", 'w') as my_zip:
#     for i in filelist:
#         my_zip.write(i)
#     my_zip.close()