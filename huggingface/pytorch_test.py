from __future__ import print_function
import torch

x = torch.randn(3, requires_grad=True)
print("x", x)

y = x * 2
print( "y = x*2", y)
i=0

#norm函数是求平方根
while y.data.norm() < 1000:
    print("y.data.norm", y.data.norm())
    y = y*2
    print("y = y*2", y)
    i=i+1

print(y)
print(i)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)