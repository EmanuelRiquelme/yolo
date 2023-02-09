import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet34, ResNet34_Weights

class Yolo(nn.Module):
    def __init__(self,grid_size = 7,num_classes = 20,bb_pred = 2):
        super(Yolo, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.bb_pred = bb_pred
        self.model = self.__get_model__()

    def __get_model__(self):
        model_ft = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        model_ft.fc = nn.Linear(512,self.grid_size**2*(self.num_classes+self.bb_pred*5))
        return model_ft

    def forward(self,batch_img):
        return self.model(batch_img).reshape(-1,self.grid_size ** 2,self.num_classes+ 5*self.bb_pred)
