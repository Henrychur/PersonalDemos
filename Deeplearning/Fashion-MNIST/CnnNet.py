import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # [1,32,32]->[64,32,32]
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(3,3),stride=1,padding=1)
        # ->[64,16,16]
        self.max_pool = nn.MaxPool2d(2)
        # ->[128,16,16]->[128,8,8]
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1)
        # ->[256,8,8]->[256,4,4]
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1)
        # ->[512,4,4]->[512,2,2]
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=1)
        # self.conv5 = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=(1,1),stride=1,padding=0)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(2048,10),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(128,32),
            # nn.ReLU(),
            # nn.Linear(32,10),
            nn.Softmax(dim=-1)
        )


    def forward(self,x):
        batch_size,_,_,_ = x.shape
        # block 1
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.bn64(x)
        x = self.relu(x)
        # block 2
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.bn128(x)
        x = self.relu(x)
        # block 3
        x = self.conv3(x)
        x = self.max_pool(x)
        x = self.bn256(x)
        x = self.relu(x)
        # block 4
        x = self.conv4(x)
        x = self.max_pool(x)
        x = self.bn512(x)
        x = self.relu(x)
        # fc
        x = x.reshape(batch_size,-1) #->[batch_size,512]
        x = self.fc(x)
        return x