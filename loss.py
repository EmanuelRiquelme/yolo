import torch
import torch.nn as nn
from utils import intersection_over_union as iou

class YoloLoss(nn.Module):
    def __init__(self, grid = 7,num_classes = 20, bb_pred = 2,coord = 5,noob = 0.5):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.grid = grid
        self.num_classes = num_classes
        self.eph = 1e-8
        self.bb_pred = bb_pred
        self.coord = coord
        self.noob = noob

    def __best_estimators__(self,gt,pred):
        bbox_1 =  pred[...,21:25]
        bbox_2 = pred[...,26:30]
        gt_bbox = gt[...,21:25]
        iou_1 = iou(gt_bbox,bbox_1).squeeze(-1)
        iou_2 = iou(gt_bbox,bbox_2).squeeze(-1)
        ious = torch.stack((iou_1,iou_2),-1)
        idx =  torch.argmax(ious,-1).unsqueeze(-1)
        return bbox_1*(1-idx)+bbox_2*idx

    def __box_loss__(self,gt,pred):
        pred_bbox = self.__best_estimators__(gt,pred)
        char = gt[...,20].unsqueeze(-1)
        bbox_loss = self.mse(gt[...,21:23],pred_bbox[...,0:2]*char)
        bbox_loss += self.mse(torch.sqrt(torch.abs(gt[...,23:25])),
                            torch.sign(pred_bbox[...,2:4])*torch.sqrt(torch.abs(pred_bbox[...,2:4]+self.eph))*char)
        return self.coord*bbox_loss


    def __obj_no_obj_loss__(self,gt,pred):
        char = gt[...,20]
        obj_loss = self.mse(char,pred[...,20]*char)
        obj_loss += self.mse(char,pred[...,25]*char)
        no_obj_loss =  self.mse(char*(1-char),pred[...,20]*(1-char))
        no_obj_loss +=  self.mse(char*(1-char),pred[...,25]*(1-char))
        return obj_loss + self.noob*no_obj_loss
   
    def __class_loss__(self,gt,pred):
        char = gt[...,20].unsqueeze(-1)
        return self.mse(gt[...,:20],char*pred[...,:20])

    def forward(self,gt,pred):
        pred = pred.reshape(-1,self.grid**2,(self.num_classes + self.bb_pred*5))
        loss = self.__box_loss__(gt,pred)
        loss += self.__obj_no_obj_loss__(gt,pred)
        loss += self.__class_loss__(gt,pred)
        return loss/pred.size(0)
