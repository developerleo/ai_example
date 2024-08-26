import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        #self.targets = torch.LongTensor(targets)
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            #x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 2, 0))
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 0))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


# Let's create 10 RGB images of size 128x128 and 10 labels {0, 1}
'''data = list(np.random.randint(0, 255, size=(3640, 1, 16, 16)))
targets = list(np.random.randint(2, size=(3640)))

transform = transforms.Compose([transforms.ToTensor()])
dataset = MyDataset(data, targets, transform=transform)
dataloader = DataLoader(dataset, batch_size=5)
print(11111)'''