import random
import torch
from d2l import torch as d2l
import numpy as np
from torch.utils import data
def synthetic_data(w,b,num_example):
    x=torch.normal(0,1,(num_example,len(w)))
    y=torch.matmul(x,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))

true_w=torch.tensor([2,3.4])
ture_b=4.2
features,labels=synthetic_data(true_w,ture_b,100)
def load_array(data_arrays,batch_size,is_train=True):
    "构造一个PYTORCH数据迭代器"
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=True)

batch_size=10
data_iter=load_array((features,labels),batch_size)
next(iter(data_iter))

from torch import nn
net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss=nn.MSELoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

num_epochs=3
for epoch in range(num_epochs):
    for x,y in data_iter:
        l=loss(net(x),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print("epoch", epoch + 1, "loss", l)