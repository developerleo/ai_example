import torch
from torch import nn

import ImgHandle as IMG
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from datasets import Dataset
import torchvision
from torch.utils.data import DataLoader
import tensorboard
from torch.utils.tensorboard import SummaryWriter


# https://blog.csdn.net/weixin_45488428/article/details/129200612
class TrainModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        self.model.add_module('Conv2d_1',
                              Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=(2, 2)))
        self.model.add_module('MaxPool2d_1', MaxPool2d(kernel_size=2))
        self.model.add_module('Conv2d_2', Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2))
        self.model.add_module('MaxPool2d_2', MaxPool2d(kernel_size=2))
        self.model.add_module('Conv2d_3', Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2))
        self.model.add_module('MaxPool2d_3', MaxPool2d(kernel_size=2))
        self.model.add_module('Flatten', Flatten())
        self.model.add_module('Linear_1', Linear(64 * 4 * 4, 64))
        self.model.add_module('Linear_2', Linear(64, 10))
        for (name, module) in self.model.named_children():
            module.register_forward_hook(self.forward_hook)

        '''self.model = nn.Sequential()
        self.model.add_module('Conv2d_1',
                              Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=(1, 1)))
        self.model.add_module('MaxPool2d_1', MaxPool2d(kernel_size=2))
        self.model.add_module('Conv2d_2', Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))
        self.model.add_module('MaxPool2d_2', MaxPool2d(kernel_size=2))
        self.model.add_module('Conv2d_3', Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))
        self.model.add_module('MaxPool2d_3', MaxPool2d(kernel_size=2))
        self.model.add_module('Flatten', Flatten())
        self.model.add_module('Linear_1', Linear(64, 3640))
        self.model.add_module('Linear_2', Linear(3640, 10))
        for (name, module) in self.model.named_children():
            module.register_forward_hook(self.forward_hook)'''

    def forward(self, x):
        x = self.model(x)
        return x

    def forward_hook(self, module, input, output):
        print("model: {} input: {} output: {}".format(module.__class__.__name__, input[0].shape, output.shape))




#input = torch.ones((64, 1, 32, 32))
#input = torch.ones((3640, 1, 16, 16))
#net = TrainModule()
#output = net(input)
#print("output shape : ", output.shape)

net = TrainModule()
train_data=torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
# 查看数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练数据集长度为：{}'.format(train_data_size))
print('测试数据集长度为：{}'.format(test_data_size))

train_dataloader = DataLoader(train_data,batch_size=16)
test_dataloader = DataLoader(test_data,batch_size=16)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器,SGD为随机梯度下降法
# 传入需要梯度下降的参数以及学习率，1e-2等价于0.01
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数【电脑性能有限，只打个样例】
epoch = 2
writer = SummaryWriter('./logs')
for i in range(epoch):
    print('------第{}轮训练开始------'.format(i + 1))
    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        # 将图片传入神经网络后得到输出结果
        outputs = net(imgs)
        # 将输出结果与原标签进行比对，计算损失函数
        loss = loss_fn(outputs, targets)
        # 在应用层面可以简单理解梯度清零，反向传播，优化器优化为三个固定步骤
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播，更新权重
        loss.backward()
        # 对得到的参数进行优化
        optimizer.step()
        total_train_step += 1
        # 为避免打印太多，训练100次才打印1次
        if total_train_step % 100 == 0:
            # loss.item()作用是把tensor转为一个数字
            print('------训练次数：{},Loss:{}------'.format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    # with的这段语句可以简单理解为提升运行效率
    with torch.no_grad():
        # 拿测试集中的数据来验证模型
        for data in test_dataloader:
            imgs, targets = data
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # agrmax(1)是将tensor对象按行看最大值下标进行存储，此处是数字图像，因此最大值下标实则就是我们的预测值
            # 此处是拿标签进行验证，统计预测正确的概率，方便后边计算正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print('整体测试集上的Loss:{}'.format(total_test_loss))
    print('整体测试集上的正确率:{}'.format(total_accuracy / test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    total_test_step += 1
    # 将每轮训练的模型都进行保存，方便以后直接调用训练完毕的模型
    torch.save(net, 'tarin_{}.pth'.format(total_test_step))

writer.close()


'''data, label = IMG.img_handle()
# data 是 3640 * 16 * 16 的图片数据
input1 = torch.ones((3640, 1, 16, 16))
net = TrainModule()
output = net(input)'''


