import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        #다크넷에서 계산된 결과는 predictions에 담김
        #앞서 모델 출력 정의된 대로, predictions 는 배치, 그리드*그리드(클래스+박스*5)
        #타겟은 라벨링 데이터로 데이터셋 로더에서 같은 테젓 차원으로 정의되어 있음
        #프레딕션 결과는 모델의 가중치가 곱해진 결과
        predictions = predictions.reshape(-1, self.S, self.S, self.C+self.B*5)

        #계산된 박스1과 라벨링된 박스2 사이의 iou계산
        #계산됟어야 할 박스는 2개음로 각가의 계산
        iou_b1 = intersection_over_union(predictions[...,21:25], target[...,21:25])
        iou_b2 = intersection_over_union(predictions[...,26:30], target[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) #b1,2중 최대 iou를 가진 맥스익덱스를 얻ㅇ디위해 언스퀴즈후 0차원기준으로 연접

        #두 박스중 iou 큰 박스를 취함
        iou_maxes, bestbox = torch.max(ious, dim=0) #0차원을 기준으로 최대 iou를 가진 iou max값과 bestbox인덱스를 얻고 16,7,7,1
        exists_box = target[..., 20].unsqueeze(3) #각 격자의 오브젝트니스 얻음 16,7,7 이를 예측된 박스중 최대 iou박스를 어딕위해 베스트박스 차원과 ㅏㅁㅈ춰야함 그러므로 16,7,7,1

        #손실을 줄일 기준 데이터인 타겟 데이터에 객체가 존재하면 타겟데이터와 가장 높은 iou를 가진 박스인 박스프레딕션을 얻음
        box_predictions = exists_box*(
            (
                bestbox*predictions[..., 26:30]
                + (1-bestbox) * predictions[..., 21:25]
            )
        )

        #타겟 데이터의 박스를 얻음 오브젝트니스가 없다면 박스 타겟은 0은 이되어 아래코드는 계산되지 않음
        box_targets = exists_box*target[..., 21:25]

        #loss 함수 정의 각 로스는 샘플단위로 계산되도록 변경된 후 계산

        #bbox w h loss = mse(pred box, target box)
        #이미지 데이터 입력된 모델을 통해 얻은 박스의 높이 너비를 절대값 루트 처리후 원래 박스 프레딕션값에 대한 +/- 부호 적용
        box_predictions[...,2:4] = torch.sign(box_predictions[..., 2:4])*torch.sqrt(
            torch.abs(box_predictions[..., 2:4]+1e-6)

        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4]) #타겟 bbox wh 루트

        #박스에 대한 mse계산 이를 위해 ㅂ먼저 박스 프레딕션 텐서 16,7,7,4,를 첫차우너붵 마지막차원 -2까지 평활화햇 784, 4로 만들고 각 784의 격자별로 종속된 박스와 타겟박스와의 차이값을 mse계산
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        #예측 계산된 박스 두개중 라벨링된 박스와 iou가 장높은 오브젝트니스 값인 프레드박스를 얻음
        pred_box = (
            bestbox*predictions[...,25:26] + (1-bestbox)*predictions[..., 20:21]
        )

        #iou 가장 높은 프레드박스 오브젝트니스값과 라벨링된 타겟데이터의 오브젝트니스값과의 mse를 통한 오브젝트 로스
        object_loss = self.mse(
            torch.flatten(exists_box*pred_box),
            torch.flatten(exists_box*target[..., 20:21])
        )

        #yolo손실함수에 정의된 nmo objec loss 계산 앞서 objecteness계산한것과 반대로 계산하므로 -1값을 취하여 계산된 예측, 타겟 ㅇ르 평활해 16,49로 변환후 mse
        no_object_loss =self.mse(
            torch.flatten((1-exists_box)*predictions[...,20:21], start_dim=1),
            torch.flatten((1-exists_box)*target[...,20:21], start_dim=1),
        ) #첫번째 계산된 bbox의 로스

        no_object_loss += self.mse(
            torch.flatten((1-exists_box)*predictions[..., 25:26], start_dim=1),
            torch.flatten((1-exists_box)*target[..., 20:21], start_dim=1),
        ) #두번째 계산된 bbox의 로스

        #클래스 loss함수정의 mse를 위해 프레딕션 :20 16,7,7,20을 자막차원 추 -2까지 평활하햇 784,20으로 만듬ㄻ
        class_loss = self.mse(
            torch.flatten(exists_box*predictions[...,20], end_dim=-2),
            torch.flatten(exists_box*target[...,20], end_dim=-2),
        )

        #손실값 전체 합함, 각 손실함수의 가중치를 고려해 바운딩박스 좌표손실과 노오브제 로스에 람다값 곱
        loss = (
            self.lambda_coord*box_loss
            +object_loss
            +self.lambda_noobj*no_object_loss
            +class_loss
        )
        return loss

