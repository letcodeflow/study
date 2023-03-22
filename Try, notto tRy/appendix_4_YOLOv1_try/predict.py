import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from loss import YoloLoss
curd = os.path.dirname(os.path.abspath(__file__))+'\\'
from PIL import Image
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PredictCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img

if __name__ == '__main__':
    img_fnames = ['','','',]

    model = Yolov1(split_size = 7, num_boxes=2, num_classes=20).to(DEVICE)
    model.load_state_dict(torch.load(curd+'yolo_last.pth'))

    for img_fname in img_fnames:
        image = Image.open(curd+img_fname)

        transform = PredictCompose([transforms.Resize((448,448)), transforms.ToTensor()])
        x = transform(image)

        c, w, h = x.shape
        input = torch.reshape(x, (1,c,w,h))
        input = input.to(DEVICE)


        pred = model(input)

        batch_bboxes = cellboxes_to_boxes(pred)

        for bboxes in batch_bboxes:
            bboxes = non_max_suppression(bboxes, iou_threshold=.5, threshold=.4, box_format='midpoint')

            plot_image(x.permute(1,2,0).to('cpu'), bboxes)
