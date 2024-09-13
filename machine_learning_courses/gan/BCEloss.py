import torch
from torch import autograd
#手动模拟Pytorch中的BCELoss是怎么计算的

# 随机值 当做输入
input = autograd.Variable(torch.tensor([
    [1.9072, 1.1079, 1.4906],
    [-0.6584, -0.0512, 0.7608],
    [-0.0614, 0.6583, 0.1095]
], requires_grad=True))

print("input:", input)
print('-' * 80)

from torch import nn

m = nn.Sigmoid()
print(m(input))
print('-' * 100)

#预测数来的标签值
target = torch.FloatTensor([[0, 1, 1], [1, 1, 1], [0, 0, 0]])
print(target)
print('-' * 100)

import math

r11 = 0 * math.log(0.8707) + (1 - 0) * math.log(1 - 0.8707)
r12 = 1 * math.log(0.7511) + (1 - 1) * math.log(1 - 0.7511)
r13 = 1 * math.log(0.8162) + (1 - 1) * math.log(1 - 0.8162)

r21 = 1 * math.log(0.3411) + (1 - 1) * math.log(1 - 0.3411)
r22 = 1 * math.log(0.4872) + (1 - 1) * math.log(1 - 0.4872)
r23 = 1 * math.log(0.6816) + (1 - 1) * math.log(1 - 0.6816)

r31 = 0 * math.log(0.4847) + (1 - 0) * math.log(1 - 0.4847)
r32 = 0 * math.log(0.6589) + (1 - 0) * math.log(1 - 0.6589)
r33 = 0 * math.log(0.5273) + (1 - 0) * math.log(1 - 0.5273)

r1 = -(r11 + r12 + r13) / 3
r2 = -(r21 + r22 + r23) / 3
r3 = -(r31 + r32 + r33) / 3

bceloss = (r1 + r2 + r3) / 3
print("手动计算出的bceloss:", bceloss)
print('-' * 100)

#需要自己做变换
loss = nn.BCELoss()
print('touch nn 计算出的bce loss', loss(m(input), target))

#不需要自己做变换
loss = nn.BCEWithLogitsLoss()
print('touch nn 计算出的bce loss', loss(input, target))