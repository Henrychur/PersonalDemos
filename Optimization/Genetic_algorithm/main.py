import matplotlib.pyplot as plt
from GA import GA
from Data import DataMaker

def fig(cities,order):
    #绘制路线图
    X = []
    Y = []
    for i in order:
        x = cities[i][0]
        y = cities[i][1]
        X.append(x)
        Y.append(y)
    plt.scatter(X,Y)
    plt.fill(X,Y,'-o',fill=False)
    plt.title("satisfactory solution of TS")
    plt.show()


def main():
    datamaker = DataMaker(10)
    cities = datamaker.get_data()
    # datamaker.show()
    ga = GA(city_num=10, population=10,max_iters=100,pc=0.5,pm=0.01)
    order = ga.fit(cities)
    fig(cities, order)
    log = ga.get_log()
    plt.plot(log)
    plt.show()

if __name__ == "__main__":
    main()