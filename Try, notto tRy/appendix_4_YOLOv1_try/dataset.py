import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset): #데이터셋 파생클래스
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ): #생성자
    #라벨링 데이터 파일 읽음
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir #학습데이터
        self.label_dir = label_dir #라벨데이터
        self.transform = transform #데이터 변환기
        self.S = S
        self.B = B
        self.C = C

    def __len__(self): #오버라이딩
        return len(self.annotations) #학습 데이터 갯수

    def __getitem__(self, index): #오버라이딩
        #라벨 파일 폴더에서 index에 대한 라벨 파일명 경로
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        #라벨 파일 파싱해 bbox 리스트 획득
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                #클래스 라벨, x,y,폭,높이
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace('\n','').split()
                ]

                #경계박스 리스트 추가
                boxes.append([class_label, x, y, width, height])

        #이미지 파일명 
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes) #학습되도록 bbox 텐서 변환

        #이미지 448 448해상도 변환 후 텐서 변환
        #경계박스 형식 class, cx, cy, w, h
        if self.transform:
            image, boxes = self.transform(image, boxes)

        #로딩된 각 bbox별로 학습 데이터를 모델 예측결과와 비교해 손실값 계산될 수 있도록
        #모델 계산 출력과 동일한 텐서 차원으로 맞춰주어, 데이터를 라벨 매트릭스에 넣어줌
        label_matrix = torch.zeros((self.S, self.S, self.C+5*self.B)) #격 7x7, class수
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            #x,y,는 정규화된 bbox중심좌표값, 격자갯수곱해서 xy대응하는 격자 index 곱함
            i, j = int(self.S*y), int(self.S*x)
            #격자 인덱스에 대한 x,y의 상대좌표르 구함, 이를 위해, 격자좌표계로 변환된 x,y에서 격자 인덱스 i,j를 빼줌
            x_cell, y_cell = self.S*x - j, self.S*y-i

            #정규화된 bbox의 width height를 격자좌표계로 변환
            width_cell, height_cell = (
                width*self.S,
                height*self.S,
            )

            #라벨링된 bbox에 x,y에 대응하는 각 격자 i, j의 텐서에 bbox, class 값을 설정함
            #이를 위해 먼저 라벨 매트릭싀 오브젝트값이 이미 할당돼있는지 확임
            if label_matrix[i,j,20] ==0: #해당 i,j 격자에 bbox 객체정보가 할당된 것이 없으면
                #i, j에 bbox와 객체가 있으므로 objectennes값은 1
                label_matrix[i,j,20] = 1

                #객체 bbox값을 격자 i,j 의 상대좌표 x_cell, y_cell로 설정, 격자좌표계로 bbox 폭 너비 설정
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                #객체 bbox값은 21-25 까지 텐서로 저장
                label_matrix[i,j, 21:25] = box_coordinates
                
                #one hot 인코딩으로 클래스라벨에 해당하는 벡터요소 부분에 1로 설정
                label_matrix[i,j, class_label] = 1

        return image, label_matrix

