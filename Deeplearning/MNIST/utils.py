import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def choose_data(images, labels,data_num):
    if data_num > 10:
        return images[:data_num], labels[:data_num]
    else:
        bucket = [0 for _ in range(10)]
        pre_images, pre_labels = [], []
        for image,label in zip(images, labels):
            if bucket[label] == 0:
                bucket[label]=1
                pre_images.append(image)
                pre_labels.append(label)
            else:
                if min(bucket)==1:
                    break
        return pre_images, pre_labels


def show_img():
    images, labels =  load_mnist("/data/pcqiu/Datas/minist","train")
    for i in range(10):
        plt.subplot(2,5,1+i)
        title = u"label:" + str(labels[i])
        plt.title(title)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        plt.imshow(images[i].reshape(28,28),cmap='gray')
    plt.savefig("tmp.png")
    plt.show()

# test
if __name__=="__main__":
    images, labels =  load_mnist("/data/pcqiu/Datas/minist","train")
    img, lab = choose_data(images,labels,10)
    print(len(img))
    print(len(lab))