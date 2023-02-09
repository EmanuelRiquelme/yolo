import torch
import torch.optim as optim
from model import Yolo
from dataset import VOC_Dataset
from torch.utils.data import DataLoader
from loss import YoloLoss
import os
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from utils import save_model,load_model
import argparse

#parse batch_size, init lr, max_lr, epochs,folder training data
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str,
                        default='train',
                        help='root directory of dataset')
parser.add_argument('--BS', type=int, default=64,
                        help='Batch size')
parser.add_argument('--fine_tune', type=bool, default=False,
                        help='retrain the model')
parser.add_argument('--epochs', type=int, default=135,
                        help='number of epochs')
parser.add_argument('--min_lr', type=float, default=1e-3,
                        help='lowest learning rate')
parser.add_argument('--max_lr', type=float, default=1e-2,
                        help='highest learning rate')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
traing_data = VOC_Dataset(args.root_dir)
train_dataloader = DataLoader(traing_data, batch_size = args.BS, shuffle=True)
model = Yolo().to(device)
load_model(model) if args.fine_tune else None
optimizer = optim.SGD(model.parameters(), lr=args.min_lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs)
loss = YoloLoss()
writer = SummaryWriter()

def train(data_loader = train_dataloader,optimizer = optimizer,model = model,loss_fn =loss,device = device,epochs = args.epochs):
    for epoch in  (t := trange(epochs)):
        temp_loss = []
        it = iter(data_loader)
        for _ in range(len(data_loader)):
            input,target = next(it)
            input,target = input.to(device),target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(target,output)
            temp_loss.append(loss.item())
            loss.backward()
            t.set_description("Loss: %.2f" % loss)
            optimizer.step()
            scheduler.step()
        writer.add_scalar('Loss',sum(temp_loss)/len(temp_loss),epoch)
    save_model(model)

if __name__ == '__main__':
    train()
