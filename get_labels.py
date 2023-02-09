from bs4 import BeautifulSoup
import numpy as np
import re
from utils import create_grids,get_mid_points
classes = [
"person","bird","cat","cow",
"dog","horse","sheep","aeroplane",
"bicycle","boat","bus","car","motorbike",
"train","bottle","chair","diningtable",
"pottedplant","sofa","tvmonitor"
]
class Labels:
    def __init__(self,img_name,root_dir,grid_size = 7,classes = classes,b = 2):
        self.name = img_name
        self.label = f'{root_dir}/Annotations/{img_name}.xml' 
        self.grid_size = grid_size
        self.size = self.__get_size__()
        self.classes = classes
        self.b = b

    def __remove_tags__(self,key):
        format = re.compile('<.*?>')
        try:
            return float(re.sub(format, '',str(key)))
        except:
            return re.sub(format, '',str(key))

    def __get_size__(self):
        with open(self.label, 'r') as file:
            file = BeautifulSoup(file, 'xml')
            size_tag = file.find('size')
        height,width = size_tag.find('height'),size_tag.find('width')
        height,width = self.__remove_tags__(height),self.__remove_tags__(width)
        return list((height,width))

    def __get_bbox__(self):
        points_tag_names = ['xmin','ymin','xmax','ymax']
        with open(self.label, 'r') as file:
            file = BeautifulSoup(file, 'xml')
            objects_det = file.find_all('object')
        bboxes_tag_format =  [object_det.find('bndbox') for object_det in objects_det]
        bboxes = []
        for bbox in bboxes_tag_format:
            bboxes.append([self.__remove_tags__(bbox.find(points)) for points in points_tag_names])
        bboxes =  np.array(bboxes)
        bboxes = get_mid_points(bboxes,sizes = self.size)
        return bboxes

    def __intersection_bboxes__(self):
        bboxes = self.__get_bbox__()
        grids = create_grids(self.grid_size)
        indeces = []
        for bbox in bboxes:
            x_axis = (bbox[0]<=grids[...,2])&(bbox[0]>=grids[...,0])
            y_axis = (bbox[1]<=grids[...,3])&(bbox[1]>=grids[...,1])
            index = np.where(x_axis&y_axis)
            indeces.append(index)
        return np.hstack(indeces).flatten()

    def __one_hot_vector__(self,labels):
        hot_labels = []
        for label in labels:
            hot_label = np.zeros(len(self.classes))
            hot_label[label] = 1
            hot_labels.append(hot_label)
        return np.array(hot_labels)

    def __get_classes__(self):
        num_classes = np.arange(len(self.classes))
        dict_classes = {}
        for value,key in zip(num_classes,self.classes):
            dict_classes[key] = value
        with open(self.label, 'r') as file:
            file = BeautifulSoup(file, 'xml')
            objects_det =  file.find_all('object')
        class_labels = [self.__remove_tags__(object_det.find('name')) for object_det in objects_det]
        class_labels =  [dict_classes[label] for label in class_labels]
        return self.__one_hot_vector__(class_labels)

    def get_labels(self):
        labels = np.zeros([self.grid_size**2,len(self.classes)+5*self.b])
        grid = create_grids(self.grid_size)
        for idx,class_hot_vector,bbox in zip(self.__intersection_bboxes__(),self.__get_classes__(),self.__get_bbox__()):
            labels[idx][:20] = class_hot_vector
            labels[idx][20] = 1
            labels[idx][21:23] = (bbox[:2]-grid[idx][:2])*self.grid_size
            labels[idx][23:25] =  bbox[2:]*self.grid_size
        return labels
