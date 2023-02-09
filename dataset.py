from PIL import Image
import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
from get_labels import Labels

class VOC_Dataset(Dataset):
    def __init__(self, root_dir = 'sample',transform = None):
      self.root_dir = root_dir
      self.transforms = transforms if transform else transforms.Compose([
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.ColorJitter(saturation=.5),
                    transforms.RandomRotation(degrees = .2),
                    transforms.Resize((448,448)),
                    ])

    def __name_files__(self):
        return [file_name[:-4] for file_name in os.listdir(f'{self.root_dir}/Images')]

    def __len__(self):
      return len(self.__name_files__())

    def __getitem__(self, idx):
       file_name = self.__name_files__()[idx]
       img = Image.open(f'{self.root_dir}/Images/{file_name}.jpg')
       return self.transforms(img),torch.tensor(Labels(img_name = file_name,root_dir = self.root_dir).get_labels())
