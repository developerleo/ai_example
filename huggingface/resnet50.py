import torch
import timm
import torch.optim as optim
import torch.nn as nn

model = timm.create_model('resnet50', pretrained=True)
#print(model)

# Define your optimizer with weight decay
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)

loss_function = nn.CrossEntropyLoss()

