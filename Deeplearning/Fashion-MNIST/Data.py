from Config import config
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt

def get_dataloader(mode):
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    
    if mode == "train":
        trainset = datasets.FashionMNIST('./dataset/', download=True, train=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True,num_workers=config.num_workers)
        return trainloader
    elif mode == "test":
        testset = datasets.FashionMNIST('./dataset/', download=True, train=False, transform=transform)
        testloader = DataLoader(testset, batch_size=config.batch_size, shuffle=True,num_workers=config.num_workers)
        return testloader

def show_example_data():
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
        trainset = datasets.FashionMNIST('./dataset/', download=True, train=True, transform=transform)
        cnt = 0
        for img,label in trainset:
            if label == cnt:
                plt.subplot(2,5,label+1)
                plt.imshow(img[0,...])
                cnt += 1
            if cnt == 10:
                break
        plt.savefig("tmp.png")

if __name__ == "__main__":
    show_example_data()
