import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np
import seaborn as sns
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
train_data=pd.read_csv("competition2/train.csv")
test_data=pd.read_csv("competition2/test.csv")

leaves_labels = sorted(list(set(train_data['label'])))
n_classes = len(leaves_labels)

class_to_num=dict(zip(leaves_labels,range(n_classes)))
num_to_class={v:k for k,v in class_to_num.items()}
#构建序号与名称的双向字典

class LeavesData(Dataset):
    def __init__(self,csv_path,file_path,mode="train"):
        self.mode=mode
        self.file_path=file_path
        self.data_info=pd.read_csv(csv_path,header=None)
        self.data_len=len(self.data_info.index)-1
        if mode=="train":
            self.train_image=np.asarray(self.data_info.iloc[1:self.data_len,0])
            self.train_label=np.asarray(self.data_info.iloc[1:self.data_len,1])
            self.image_arr=self.train_image
            self.label_arr=self.train_label
        elif mode=="test":
            self.test_image=np.asarray(self.data_info.iloc[1:,0])
            self.image_arr=self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
                  .format(mode, self.real_len))
        #该步完成对数据集的读取与划分
    def __getitem__(self, index):
        single_image_name=self.image_arr[index]
        img_as_img=Image.open(self.file_path+single_image_name)
        if self.mode=="train":
            transfrom=transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            transfrom = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])

        img_as_img=transfrom(img_as_img)

        if self.mode=="test":
            return img_as_img
        else :
            label=self.label_arr[index]
            number_label=class_to_num[label]
            return img_as_img,number_label
        #通过序号获得图片
    def __len__(self):
        return self.real_len

train_path="competition2\\train.csv"
test_path="competition2\\test.csv"
img_path='D:/pycharm/lynew/competition2/'

train_dataset=LeavesData(csv_path=train_path,file_path=img_path,mode='train')
test_dataset=LeavesData(csv_path=test_path,file_path=img_path,mode="test")

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=0)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=True,num_workers=0)

print(train_loader)

if __name__=='__main__':
    def im_convert(tensor):
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image.clip(0, 1)

        return image
    fig = plt.figure(figsize=(20, 12))
    columns = 4
    rows = 2

    dataiter = iter(train_loader)
    inputs, classes = dataiter.next()
    for idx in range (columns*rows):
        ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
        ax.set_title(num_to_class[int(classes[idx])])
        plt.imshow(im_convert(inputs[idx]))
    #plt.show()

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

class Residual(nn.Module):
    def __init__(self,
                 input_channels,num_channels,
                 use_1x1conv=False,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,
                             kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,
                             kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(input_channels,num_channels,
                                 kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)
    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return F.relu(Y)

b1=nn.Sequential(nn.Conv2d(3,64,kernel_size=7,padding=3,stride=2),
                 nn.BatchNorm2d(64),nn.ReLU(),
                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

def resnet_block(input_channels,num_channels,
                 num_residuals,first_block=False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels,num_channels,
                                use_1x1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk

b2=nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3=nn.Sequential(*resnet_block(64,128,2))
b4=nn.Sequential(*resnet_block(128,256,2))
b5=nn.Sequential(*resnet_block(256,512,2))

def get_net():
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(512, 176))
    return net.to(get_device())

loss=nn.CrossEntropyLoss()
net = get_net()
net=net.to(device)
def log_rmse(net,features,labels):
    clipped_preds=torch.clamp(net(features),1,float('inf'))
    rmse=torch.sqrt(loss(torch.from_numpy(clipped_preds.cpu().detach().numpy()),
                         torch.from_numpy(labels.cpu().detach().numpy())))
    return rmse.item()


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]

def train(net,train_features,train_labels,valid_features,valid_labels,
          num_epochs,learning_rate,weight_decay,batch_size,device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    train_ls,valid_ls=[],[]
    train_iter=d2l.load_array((train_features,train_labels),batch_size)
    valid_iter=d2l.load_array((valid_features,valid_labels),batch_size)
    optimizer=torch.optim.Adam(net.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                              legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), train_features.shape[0]
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i,(X,y) in enumerate(train_iter):
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if train_l<2.5:
                animator.add(epoch + i + 1,
                              (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, valid_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    train_ls.append(log_rmse(net,train_features,train_labels))
    if valid_labels is not None:
        valid_ls.append(log_rmse(net,valid_features,valid_labels))
    return train_ls,valid_ls


def get_k_flod_data(k,i,X,y):
    assert k>1
    fold_size=X.shape[0]//k
    X_train,y_train=None,None
    X_valid,y_valid=None,None
    for j in range (k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        X_part,y_part=X[idx,:],y[idx]
        if j==i:
            X_valid,y_valid=X_part,y_part
        elif X_train is None:
            X_train,y_train=X_part,y_part
        else:
            X_train=torch.cat([X_train,X_part],0)
            y_train=torch.cat([y_train,y_part],0)
    return X_train,y_train,X_valid,y_valid

def k_fold(k,X_train,y_train,num_epochs,learning_rate,
           weight_decay,batch_size):
    train_l_sum,valid_l_sum=0,0
    for i in range(k):
        data=get_k_flod_data(k,i,X_train,y_train)

        train_ls,valid_ls=train(net,*data,num_epochs,
                                learning_rate,weight_decay,batch_size,get_device())
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


dataiter = iter(train_loader)
train_features,train_labels = dataiter.next()

train_features,train_labels =train_features .to(device), train_labels.to(device)

k, num_epochs, lr, weight_decay, batch_size = 4, 100, 0.001, 0, 128
train_l, valid_l = k_fold(k,train_features,train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

