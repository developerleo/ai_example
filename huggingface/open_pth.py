import torch
content = torch.load('fine-tuned-bert.pth')
print('keys\n' , content.keys())
