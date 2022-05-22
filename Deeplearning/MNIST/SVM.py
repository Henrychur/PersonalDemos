from utils import load_mnist,choose_data
from sklearn.svm import SVC
import numpy as np
import time

def SVM_train(data_num):
    model = SVC(C=1,kernel="rbf",max_iter=1000)
    train_X,train_y = load_mnist("/data/pcqiu/Datas/minist","train")
    train_X,train_y = choose_data(train_X,train_y,data_num)
    test_X,test_y = load_mnist("/data/pcqiu/Datas/minist","t10k")
    model.fit(train_X,train_y)
    train_acc = model.score(train_X,train_y)
    test_acc = model.score(test_X,test_y)
    print("train acc: ",train_acc," test acc: ",test_acc)
    return train_acc,test_acc

class SmallSampleSVM():
    # This is a class for test an algorithm for classify MNIST with SVM
    # only 10 labeled data is used
    # other data has no label
    def __init__(self,data_num):
        X, y = load_mnist("/data/pcqiu/Datas/minist","train")
        self.test_X,self.test_y = load_mnist("/data/pcqiu/Datas/minist","t10k")
        self.origin_X, self.origin_y = choose_data(X,y,data_num)
        self.no_label_X = X[:5000]
        self.svm = SVC(C=1,kernel="rbf")

    def fit(self):
        tmp_y = None
        tmp_X = None
        for i in range(5):
            if i ==0:
                self.svm.fit(self.origin_X,self.origin_y)
            else:
                self.svm.fit(np.concatenate((self.origin_X,tmp_X),axis=0),np.concatenate((self.origin_y,tmp_y),axis=0))
            tmp_X = self.no_label_X
            tmp_y = self.svm.predict(self.no_label_X)
        score = self.svm.score(self.test_X,self.test_y)
        print("test acc: ",score)
        return score

if __name__=="__main__":
    scores = []
    for data_num in [10,100,500,1000,5000]:
        small_sample_svm = SmallSampleSVM(data_num)
        score = small_sample_svm.fit()
        scores.append(score)
    print(scores)
