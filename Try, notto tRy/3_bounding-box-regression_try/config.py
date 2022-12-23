import os

#현재 모듈의 폴더
curd = os.path.dirname(os.path.abspath(__file__)) +'\\'
# print('현재폴더',curd)
# 현재폴더 d:\OneDrive - 한국방송통신대학교\1_total_beat\1208 yolo\fast_yolo\4_object_detection\3_bounding-box-regression_try\
# print(os.path.abspath(__file__))
# d:\OneDrive - 한국방송통신대학교\1_total_beat\1208 yolo\fast_yolo\4_object_detection\3_bounding-box-regression_try\config.py
#하위 폴더의 데이터셋 폴더 이미지와 csv를 로딩
BASE_PATH = curd + 'dataset'
IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images'])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, 'airplanes.csv'])
# print(BASE_PATH)
# # d:\OneDrive - 한국방송통신대학교\1_total_beat\1208 yolo\fast_yolo\4_object_detection\3_bounding-box-regression_try\dataset
# print(IMAGES_PATH)
# d:\OneDrive - 한국방송통신대학교\1_total_beat\1208 yolo\fast_yolo\4_object_detection\3_bounding-box-regression_try\dataset\images
#출력 폴더 설정
BASE_OUTPUT = curd + 'output'
print(BASE_OUTPUT)
# d:\OneDrive - 한국방송통신대학교\1_total_beat\1208 yolo\fast_yolo\4_object_detection\3_bounding-box-regression_try\output

#학습 후 저장될 모델 파일명 설정
#학습 결과 plot.png 테스트 이미지들
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'detector.h5'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plot.png'])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, 'test_images.txt'])

#학습율 학습 에폭, 배치크기
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32
