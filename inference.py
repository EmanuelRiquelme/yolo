import torch
from PIL import Image
import torchvision
from model import Yolo
import os
import numpy as np
from torchvision.utils import draw_bounding_boxes
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from utils import create_grids,get_mid_points,midpoint_to_bbox
classes = [
"person","bird","cat","cow",
"dog","horse","sheep","aeroplane",
"bicycle","boat","bus","car","motorbike",
"train","bottle","chair","diningtable",
"pottedplant","sofa","tvmonitor"
]

class inference:
    def __init__(self,file_name,root_dir = 'sample/Images',transform = None,grid_size = 7,confidence_threshold = .4,classes = classes,
            model_name = 'yolo'):
        self.transform = transform
        self.model_name = model_name
        self.classes = classes
        self.confidence = confidence_threshold
        self.img_dir= f'{root_dir}/{file_name}.jpg'
        self.img_size = Image.open(self.img_dir).size
        self.img = self.__load_img_tensor__()
        self.grid_size = grid_size
        self.model = self.__load_model__()

    def __load_model__(self):
        model = Yolo()
        model.load_state_dict(torch.load(f'{os.getcwd()}/{self.model_name}.pt'))
        return model.eval()

    def __load_img_tensor__(self):
        transforms = self.transform if self.transform else torchvision.transforms.Compose([
                    torchvision.transforms.PILToTensor(),
                    torchvision.transforms.ConvertImageDtype(torch.float),
                    torchvision.transforms.ColorJitter(saturation=.5),
                    torchvision.transforms.RandomRotation(degrees = .2),
                    torchvision.transforms.Resize((448,448)),
                    ])
        return transforms(Image.open(self.img_dir)).unsqueeze(0)

    def prediction(self):
        pred = self.model(self.img).detach().numpy().squeeze(0)
        grids = create_grids(self.grid_size)
        obj_pred_1 = pred[...,20]> self.confidence
        obj_pred_2 = pred[...,25]> self.confidence
        obj_pred = np.logical_or(obj_pred_1,obj_pred_2)
        index = np.nonzero(obj_pred_2)
        preds,grids = pred[index],grids[index]
        bboxes = []
        width,height = self.img_size
        for pred,grid in zip(preds,grids):
            if pred[20] > pred[25]:
                bbox = pred[21:25]
            else:
                bbox = pred[-4:]
            bbox /= self.grid_size
            bbox[:2] += grid[:2]
            bbox = np.hstack((np.argmax(pred[:20]),bbox))
            bboxes.append(bbox)
        bboxes = np.array(bboxes)
        bboxes[...,1] *= width 
        bboxes[...,3] *= width 
        bboxes[...,2] *= height
        bboxes[...,4] *= height 
        bboxes[...,1:] = midpoint_to_bbox(bboxes[...,1:])
        return torch.tensor(bboxes)

    def plot(self):
        transform =  torchvision.transforms.Compose([
                    torchvision.transforms.PILToTensor(),
                    ])
        colors = ["blue", "yellow",'red','green']
        labels = [self.classes[int(prediction)] for prediction in self.prediction()[...,0]]
        boxes = self.prediction()[...,1:]
        result = draw_bounding_boxes(transform(Image.open(self.img_dir)), boxes = boxes ,labels = labels,colors =  colors)
        result= torchvision.transforms.ToPILImage()(result)
        result.show()

if __name__ == '__main__':
    inference(file_name = '2007_000129').plot()
