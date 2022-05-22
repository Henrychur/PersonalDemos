from matplotlib import transforms
import matplotlib.pyplot as plt
from utils import load_mnist,choose_data
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import time

class Config():
    def __init__(self,device,num_workers,batch_size,lr,epochs,data_num):
        self.device = device
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.data_num = data_num

class MinistDataset(Dataset):
    def __init__(self,data_num=5000,kind="train"):
        super().__init__()
        self.X,self.y = load_mnist("/data/pcqiu/Datas/minist",kind)
        if kind == "train":
            self.X, self.y = choose_data(self.X,self.y,data_num)

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return torch.tensor(self.X[idx].reshape(1,28,28),dtype=torch.float32),torch.tensor(self.y[idx],dtype=torch.long)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # [1,28,28]->[20,24,24]
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=20,kernel_size=(5,5),stride=1,padding=0)
        # ->[20,12,12]
        self.max_pool = nn.MaxPool2d(2)
        # ->[50,8,8]
        self.conv2 = nn.Conv2d(in_channels=20,out_channels=50,kernel_size=(5,5),stride=1,padding=0)
        self.fc = nn.Sequential(
            nn.Linear(50*4*4,500),
            nn.ReLU(),
            nn.Linear(500,10),
            nn.Softmax(dim=-1)
        )
    def forward(self,x):
        batch_size,_,_,_ = x.shape
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = x.reshape(batch_size,-1)
        x = self.fc(x)
        return x

def LeNet_train(device,num_workers,batch_size,lr,epochs,data_num,plot=False):
    config = Config(device=device,num_workers=num_workers,batch_size=batch_size,lr=lr,epochs=epochs,data_num=data_num)
    train_dataset = MinistDataset(kind = "train",data_num=data_num)
    test_dataset = MinistDataset(kind="t10k")
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=config.batch_size,num_workers=config.num_workers)

    classifier = LeNet().to(config.device)
    optimizer = torch.optim.Adam(classifier.parameters(),lr=config.lr)
    lossf = nn.CrossEntropyLoss()
    total_train_loss = []
    total_train_acc = []
    total_test_loss = []
    total_test_acc = []
    for epoch in range(1,config.epochs+1):
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
        classifier.train()
        for X,y in train_dataloader:
            X = X.to(config.device)
            y = y.to(config.device)
            optimizer.zero_grad()
            output = classifier(X)
            loss = lossf(output,y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            train_acc += (output.argmax(1)==y).sum()
        
        classifier.eval()
        for X,y in test_dataloader:
            X = X.to(config.device)
            y = y.to(config.device)

            output = classifier(X)
            loss = lossf(output,y)

            test_loss += loss
            test_acc += (output.argmax(1)==y).sum()

        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)
        test_loss /= len(test_dataset)
        test_acc /= len(test_dataset)

        total_train_loss.append(train_loss.item())
        total_train_acc.append(train_acc.item())
        total_test_loss.append(test_loss.item())
        total_test_acc.append(test_acc.item())
        if epoch % 100==0:
            print("epoch:{},train loss:{},train acc:{},test loss: {},test acc:{}".format(epoch,train_loss,train_acc,test_loss,test_acc)) 
    
    if plot:
        plt.plot(range(1,config.epochs+1),total_train_acc,label="train accuracy")
        plt.plot(range(1,config.epochs+1),total_test_acc,label="test accuracy")
        plt.legend()
        plt.savefig("train.png")
        plt.show()
    return total_train_acc, total_test_acc

    
def CIFAR10_train(model,config,optimizer,lossf):
        print("-----------------CIFAR10 TRAIN START----------------")
        transformer = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
            transforms.Resize((28,28)),
            transforms.ToTensor()
        ])
        train_dataset = CIFAR10(root="minist/data",train=True,transform=transformer,download=True)
        train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=True)
        for epoch in range(1,1+config.epochs):
            train_acc = 0
            for X,y in train_dataloader:
                X = X.to(config.device)
                y = y.to(config.device)
                optimizer.zero_grad()
                output = model(X)
                loss = lossf(output,y)
                loss.backward()
                optimizer.step()
                train_acc += (output.argmax(1)==y).sum()
            print(epoch," train_acc:",train_acc.item()/len(train_dataset))
        print("-----------------CIFAR10 TRAIN END----------------")
        
def transfer_learning(device,num_workers,batch_size,lr,epochs,data_num,plot=False):
    config = Config(device=device,num_workers=num_workers,batch_size=batch_size,lr=lr,epochs=epochs,data_num=data_num)
    train_dataset = MinistDataset(kind = "train",data_num=data_num)
    test_dataset = MinistDataset(kind="t10k")
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,num_workers=config.num_workers,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=config.batch_size,num_workers=config.num_workers)

    classifier = LeNet().to(config.device)
    optimizer = torch.optim.Adam(classifier.parameters(),lr=config.lr)
    lossf = nn.CrossEntropyLoss()
    CIFAR10_train(classifier,config,optimizer,lossf)
    optimizer = torch.optim.Adam(classifier.parameters(),lr=config.lr)
    total_train_loss = []
    total_train_acc = []
    total_test_loss = []
    total_test_acc = []
    for epoch in range(1,config.epochs+1):
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
        classifier.train()
        for X,y in train_dataloader:
            X = X.to(config.device)
            y = y.to(config.device)
            optimizer.zero_grad()
            output = classifier(X)
            loss = lossf(output,y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss
            train_acc += (output.argmax(1)==y).sum()
        
        classifier.eval()
        for X,y in test_dataloader:
            X = X.to(config.device)
            y = y.to(config.device)

            output = classifier(X)
            loss = lossf(output,y)

            test_loss += loss
            test_acc += (output.argmax(1)==y).sum()

        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)
        test_loss /= len(test_dataset)
        test_acc /= len(test_dataset)

        total_train_loss.append(train_loss.item())
        total_train_acc.append(train_acc.item())
        total_test_loss.append(test_loss.item())
        total_test_acc.append(test_acc.item())
        if epoch % 1==0:
            print("epoch:{},train loss:{},train acc:{},test loss: {},test acc:{}".format(epoch,train_loss,train_acc,test_loss,test_acc)) 
    
    if plot:
        plt.plot(range(1,config.epochs+1),total_train_acc,label="train accuracy")
        plt.plot(range(1,config.epochs+1),total_test_acc,label="test accuracy")
        plt.legend()
        plt.savefig("train.png")
        plt.show()
    return total_train_acc, total_test_acc

if __name__=="__main__":
    total_train_acc, total_test_acc = transfer_learning(device="cuda:7",num_workers=4,batch_size=16,lr=1e-5,epochs=100,data_num=500,plot=False)
    print(total_test_acc[-1])
    # start = time.time()
    # LeNet_train(device="cuda:7",num_workers=4,batch_size=16,lr=1e-5,epochs=100,data_num=60000,plot=True)
    # end = time.time()
    # cost = end - start
    # print("{}min {}s".format(cost//60,cost-60*(cost//60)))


