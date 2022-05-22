import numpy as np
import matplotlib.pyplot as plt
import random
class DataMaker():
    def __init__(self,city_nums):
        random.seed(1)
        self.cities = np.array([[random.randint(1, 100) for j in range(2)] for i in range(city_nums)])

    def show(self):
        plt.scatter(self.cities[:,0],self.cities[:,1])
        plt.title("{} cities".format(len(self.cities)))
        plt.show()
    
    def get_data(self):
        return self.cities


if __name__ == "__main__":
    datamaker = DataMaker(10)
    datamaker.show()