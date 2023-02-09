import numpy as np
import torch
import os
def create_grids(grid_size):
    x_min = np.tile((np.arange(grid_size)),grid_size)
    y_min = (np.arange(grid_size)).repeat(grid_size)
    x_max =  np.tile((np.arange(grid_size)+1),grid_size)
    y_max = (np.arange(grid_size)+1).repeat(grid_size)
    return np.stack((x_min,y_min,x_max,y_max),axis = 1)/grid_size


def get_mid_points(bboxes,sizes = [1,1]):
    height, width= sizes
    x_min,y_min,x_max,y_max = bboxes[...,0],bboxes[...,1],bboxes[...,2],bboxes[...,3]
    x_min,x_max = x_min/width,x_max/width
    y_min,y_max = y_min/height,y_max/height
    bboxes = np.array([(x_min+x_max)/2,(y_min+y_max)/2,x_max-x_min,y_max-y_min])
    return np.transpose(bboxes)

def midpoint_to_bbox(bbox):
    x1 = bbox[..., 0]-bbox[..., 2]/2
    y1 = bbox[...,1]-bbox[...,3]/2
    x2 = bbox[...,0]+bbox[..., 2]/2
    y2 = bbox[...,1]+bbox[..., 3]/2
    return np.stack((x1,y1,x2,y2),-1)

def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2
    box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def load_model(model):
    model.load_state_dict(torch.load(f'{os.getcwd()}/yolo.pt'))
def save_model(model):
    torch.save(model.state_dict(), f'{os.getcwd()}/yolo.pt')
