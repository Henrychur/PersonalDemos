import numpy as np
import random
import copy
'''
    使用GA算法解决TSP问题
'''
class GA():
    def __init__(self,city_num,population,max_iters=100,pc=0.7,pm=0.01):
        '''
        args:
            city_num:城市数量
            population:每一次迭代中存活人口得数量
            max_iters:最大迭代次数
            pc:基因重组率
            pm:基因突变率
        '''
        # 存储信息
        self.city_num = city_num
        self.population = population
        self.max_iters = max_iters
        self.pc = pc
        self.pm = pm
        # 初始化人口，随机给定population个次序
        x = [i for i in range(city_num)]
        self.cur_individuals = []
        for _ in range(population):
            random.shuffle(x)
            self.cur_individuals.append(x)
        self.dis_log = [] # 训练日志

    def fit(self,cities):
        self.cities = cities
        self.dis_log.append(self.cal_best_res(self.cur_individuals)[0])
        for epoch in range(1, self.max_iters+1):
            if epoch % 100==0:
                print(epoch," epoch has been trained.")
            next_individuals = copy.deepcopy(self.cur_individuals)
            for i in range(self.population):
                for j in range(i,self.population):
                    if random.random() > self.pc:
                        continue
                    child = self.crossover(self.cur_individuals[i], self.cur_individuals[j])
                    child = self.mutation(child)
                    next_individuals.append(child)
            survivor = self.selection(next_individuals)
            self.cur_individuals = copy.deepcopy(survivor)
            self.dis_log.append(self.cal_best_res(self.cur_individuals)[0])
        
        idx = self.cal_best_res(self.cur_individuals)[1]
        return self.cur_individuals[idx]

    def get_log(self):
        return self.dis_log

    def crossover(self,seq1,seq2):
        child = [None]*self.city_num
        #parent1
        start_pos = random.randint(0,self.city_num-1)
        end_pos = random.randint(0,self.city_num-1)
        if start_pos>end_pos:
            start_pos, end_pos = end_pos, start_pos
        child[start_pos:end_pos+1] = seq1[start_pos:end_pos+1]
        # parent2 -> child
        list1 = list(range(end_pos+1,len(seq2)))
        list2 = list(range(0,start_pos))
        list_index = list1 + list2
        j = -1
        for i in list_index:
            for j in range(j+1,len(seq2)):
                if seq2[j] not in child:
                    child[i] = seq2[j]
                    break
        return child


    def mutation(self,seq):
        for i in range(len(seq)):
            for j in range(len(seq)):
                if random.random() < self.pm:
                    seq[i],seq[j] = seq[j],seq[i]
        return seq

    def selection(self,next_individuals):
        total_fitness = 0.0
        survivor = []
        # a tricky, we always let the best individual survive.
        _, idx = self.cal_best_res(next_individuals)
        survivor.append(next_individuals[idx])
        for individual in next_individuals:
            total_fitness += self.cal_fitness(individual)
        for i in range(self.population - 1):
            random_ = random.random()
            prob = 0.0
            for individual in next_individuals:
                prob += self.cal_fitness(individual)/total_fitness
                if random_ < prob:
                    survivor.append(individual)
                    break
        return survivor

    def cal_fitness(self,seq):
        return (100/(self.cal_total_distance(seq)+1e-6))

    def cal_best_res(self,cur_individuals):
        min_dist = np.inf
        index = None
        for idx,individual in enumerate(cur_individuals):
            if min_dist > self.cal_total_distance(individual):
                min_dist = self.cal_total_distance(individual)
                index = idx
        return (min_dist,index)
    


    def cal_total_distance(self,seq):
        dis_sum = 0
        for i in range(self.city_num):
            if i < self.city_num - 1:
                dis_sum += self.__cal_distance(self.cities[seq[i]],self.cities[seq[i+1]])
            else:
                dis_sum += self.__cal_distance(self.cities[seq[i]],self.cities[seq[0]])
        return dis_sum


    def __cal_distance(self,city1,city2):
        # 欧式距离
        return ((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)**(0.5)
    