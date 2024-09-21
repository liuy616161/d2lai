import random
import torch
from d2l import torch as d2l
import numpy as np
from torch.utils import data
def synthetic_data(w,b,num_example):
    "根据具体值构造离散数据"
    x=torch.normal(0,1,(num_example,len(w)))
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))

true_w=torch.tensor([2,3.4])
ture_b=4.2
"真正值"
features,labels=synthetic_data(true_w,ture_b,100)

d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),
                labels.detach().numpy(),
                random.randint(1,10),
                'b');
"画离散图"
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=(torch.tensor
            (indices[i:min(i+batch_size,num_examples)]))
        yield features[batch_indices],labels[batch_indices]
"构造随机数，并取出相对小的组"
batch_size=10

w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
"初始化所需变量w,b"
def linreg(x,w,b):
    return torch.matmul(x,w)+b

def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2
"计算差值/方差"
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()
"随机梯度下降SGD"

lr=0.12#学习率 步长
num_epochs=3 #轮数
net=linreg
loss=squared_loss

for epoch in range(num_epochs):
    for x,y in data_iter(batch_size,features,labels):
        l=loss(net(x,w,b),y)
        "loss"
        l.sum().backward()
        "自动求导"
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_1=loss(net(features,w,b),labels)
        "拟合差值"
        print("epoch",epoch+1,"loss",float(train_1.mean()))
"训练得到答案"
