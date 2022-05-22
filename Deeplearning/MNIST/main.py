from LeNet5 import LeNet_train
from SVM import SVM_train
from matplotlib import pyplot as plt

def main():
    data_nums= [10,100,500,1000,5000]
    SVM_train_acc,SVM_test_acc,LeNet_train_acc,LeNet_test_acc = [],[],[],[]
    for data_num in data_nums:
        print("-----------data num:{}----------------".format(data_num))
        acc_train,acc_test = SVM_train(data_num=data_num)
        SVM_train_acc.append(acc_train)
        SVM_test_acc.append(acc_test)

        acc_train,acc_test = LeNet_train(device="cuda:3",num_workers=4,batch_size=8,lr=1e-5,epochs=100,data_num=data_num)
        LeNet_train_acc.append(acc_train[-1])
        LeNet_test_acc.append(acc_test[-1])        
        print()

    print("SVM test acc:")
    print(SVM_test_acc)
    print("LeNet test acc:")
    print(LeNet_test_acc)
    plt.plot(data_nums,SVM_test_acc,label="SVM test acc")
    plt.plot(data_nums,LeNet_test_acc,label="LeNet test acc")
    for x,y in zip(data_nums,SVM_test_acc):
        plt.text(x,y,'%.2f' % y,fontdict={'fontsize':8})
    for x,y in zip(data_nums,LeNet_test_acc):
        plt.text(x,y,'%.2f' % y,fontdict={'fontsize':8})
    plt.legend()
    plt.savefig("res.png")

if __name__=="__main__":
    main()