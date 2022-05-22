from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from Config import config
from Data import get_dataloader
from CnnNet import CNN


def train():
    train_dataloader = get_dataloader("train")
    test_dataloader = get_dataloader("test")
    
    model = CNN()
    model.to(config.device)

    loss_f = CrossEntropyLoss()
    loss_f.to(config.device)

    optim = Adam(params=model.parameters(),lr=config.lr)

    writer = SummaryWriter("Fashion-MNIST/logs/v2_one_linear")

    for epoch in range(1,config.epochs):
        # train
        model.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        for imgs,labels in train_dataloader:
            imgs = imgs.to(config.device)
            labels = labels.to(config.device)
            outputs = model(imgs)
            loss = loss_f(outputs,labels)

            total_train_loss += loss
            total_train_acc += (outputs.argmax(1)==labels).sum()

            #反向传播
            optim.zero_grad()
            loss.backward()
            optim.step()
        writer.add_scalar("train/loss",total_train_loss/60000,epoch)
        writer.add_scalar("train/accuracy",total_train_acc/60000,epoch)
    
        # eval
        model.eval()
        total_test_acc=0
        total_test_loss=0.0
        for imgs,labels in test_dataloader:
            imgs = imgs.to(config.device)
            labels = labels.to(config.device)
            outputs = model(imgs)
            loss = loss_f(outputs,labels)
            total_test_acc += (outputs.argmax(1)==labels).sum()
            total_test_loss += loss
        writer.add_scalar("test/loss",total_test_loss/10000,epoch)
        writer.add_scalar("test/accuracy",total_test_acc/10000,epoch)
        
        print("{}th epoch. train_loss:{:.4f},test_loss:{:.4f},test_acc:{:.4f}"\
            .format(epoch,total_train_loss/60000,total_test_loss/10000,total_test_acc/10000))

if __name__=="__main__":
    train()