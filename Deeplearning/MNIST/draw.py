import numpy as np
import matplotlib.pyplot as plt
labels =  [10,100,500,1000,5000]
y1 = [0.5099, 0.6559, 0.8711, 0.9083, 0.9513]
y2 = [0.5423, 0.6721, 0.8968, 0.9241, 0.9538]
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(len(labels))  # x轴刻度标签位置
width = 0.25  # 柱子的宽度
# 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# x - width，x， x + width即每组数据在x轴上的位置
# 这里举例三个，若多个，往两边加减width即可
plt.bar(x - width, y1, width, alpha = 0.8, label='without extra data')
for X, Y in zip(x, y1):
    plt.text(X - 0.30, Y + 0.01, '%.3f' % Y, fontsize = 6, ha = 'center', va = 'bottom')
plt.bar(x, y2, width, alpha = 0.8, label='with extra data')
for X, Y in zip(x, y2):
    plt.text(X, Y + 0.01, '%.3f' % Y, fontsize = 6, ha = 'center', va = 'bottom')

plt.ylabel('test accuracy')
# x轴刻度标签位置不进行计算
plt.xticks(x, labels = labels)
plt.xlabel("data num")
plt.legend() #在右下角生成标签，还有center、upper，如果不设置，系统自动找到合适地方
plt.savefig("tmp.png")
plt.show()


