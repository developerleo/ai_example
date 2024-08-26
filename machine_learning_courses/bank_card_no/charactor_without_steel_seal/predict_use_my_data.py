import cv2
import torch
from torch import nn
from torchvision.transforms import transforms
from pathlib import Path
from num_dataloader import MyDataset
from torch.utils.data import DataLoader

from machine_learning_courses.bank_card_no.charactor_without_steel_seal.ImgHandle import get_predict_img
from train_ues_my_data import TrainModule



#state_dict = model.state_dict()
data, label = get_predict_img()

transform = transforms.Compose([transforms.ToTensor()])
my_predict_data = MyDataset(data, label, transform=transform)

model = torch.load('./model/tarin_50.pth')

predict_data_loader = DataLoader(my_predict_data, batch_size=16)

for data in predict_data_loader:
    imgs, targets = data
    output = model(imgs)
    print(output.argmax(1).numpy()[0])
    print(output.argmax(1).numpy()[-1])