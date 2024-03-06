import random

import cv2
import numpy as np

photo = 'test.jpg'


# 定义一个遗传算法类
class GA:
    def __init__(self, image, population):
        """
        构造函数
        :param image: 待处理的图片
        :param population: 种群中个体的数目
        """
        self.image = image
        self.population = population  # 种群个体数目
        self.length = 8  # 一条染色体的长度(0-255)
        self.species = np.random.randint(0, 256, self.population)  # 初始化种群的染色体
        self.fitness = []  # 种群个体的适应度
        self.probability = []  # 种群个体的概率
        self.variation_rate = 0.01  # 变异概率

    def fitness_function(self, chromosome):
        fitness = OTSU().otsu(self.image, chromosome)  # 计算一个染色体的适应度
        return fitness

    def sum_fitness(self):
        sum_fit = 0
        for i in range(self.population):
            sum_fit += self.fitness[i][0]
        return sum_fit

    def selection(self):
        for chromosome in self.species:  # 循环遍历种群的每一条染色体，计算保存该条染色体的适应度
            self.fitness.append((self.fitness_function(chromosome), chromosome))
        # 计算每条染色体的概率，并保存
        sum_fitness = self.sum_fitness()
        for chromosome in self.fitness:
            self.probability.append((chromosome[0] / sum_fitness, chromosome[1]))
        # 计算每个染色体的累计概率，并保存
        q = []
        t = 0
        for chromosome in self.probability:
            t += chromosome[0]
            q.append((t, chromosome[1]))
        # 通过轮盘赌法选择父代
        parents = []
        for i in range(self.population):
            prob = np.random.rand()  # 产生一个[0,1]之间的一个随机浮点数
            if prob < q[0][0]:
                parents.append(q[0][1])
            else:
                for j in range(1, self.population):
                    if q[j-1][0] < prob < q[j][0]:
                        parents.append(q[j][1])
        return parents

    # 交叉
    def crossover(self, parents):
        offspring = []
        # 从父代中选择一个父亲和一个母亲进行交叉，随机选择位置交换染色体片段
        while len(offspring) < self.population:
            father = parents[np.random.randint(0, self.population)]
            mother = parents[np.random.randint(0, self.population)]
            if father != mother and father not in offspring and mother not in offspring:
                position = np.random.randint(0, self.length)
                mask = 0
                for i in range(position):
                    mask = mask | (1 << i)
                father_cross = (father & mask) | (mother & ~mask)
                mother_cross =  (mother & mask) | (father & ~mask)
                # 对染色体的右侧进行交换
                offspring.extend([father_cross,mother_cross])
        self.species = offspring
    # 变异
    def variation(self):
        for i in range(self.population):
            if np.random.random() < self.variation_rate:  # 小于该概率则进行变异，否则保持原状
                j = np.random.randint(0, self.length)  # 随机选取变异基因位置点
                self.species[i] = self.species[i] ^ (1 << j)  # 在j位置取反

    def evolution(self):  # 进行个体的进化
        parents = self.selection()
        self.crossover(parents)
        self.variation()

    def best_threshold(self):  # 返回适应度最高的这条染色体,为最佳阈值
        fitness = []
        for chromosome in self.species:  # 循环遍历种群的每一条染色体，计算保存该条染色体的适应度
            fitness.append((self.fitness_function(chromosome), chromosome))
        fitness_sorted = sorted(fitness, reverse=True)  # 逆序排序，适应度高的染色体排前面
        for i, j in zip(fitness_sorted, range(self.population)):
            fitness[j] = i[1]
        return fitness[0]

# 大津算法
class OTSU():
    def otsu(self, image, threshold):
        image = np.transpose(np.asarray(image))  # 转置ndarray矩阵
        size = image.shape[0] * image.shape[1]  # 图片像素个数
        foreground = image < threshold
        """
        对于ndarray矩阵中的元素，当元素值小于threshold返回True
        否则返回False
        """
        sum_image = np.sum(image)  # 对图像所有像素点的值求和
        w0 = np.sum(foreground)  # 小于阈值的像素点的个数
        w1 = size - w0  # 大于等于阈值的像素点的个数
        if w1 == 0:
            return 0
        sum_foreground = np.sum(foreground)  # 目标像素点的值的和
        sum_background = sum_image - sum_foreground  # 背景像素点的值的和
        mean_foreground = sum_foreground / (w0 * 1.0)  # 目标像素点的平均灰度值
        mean_background = sum_background / (w1 * 1.0)  # 背景像素点的平均灰度值
        g = w0 / (size * 1.0) * w1 / (size * 1.0) * (mean_foreground - mean_background) * (
                mean_foreground - mean_background)
        # 类间方差
        return g


def transition(threshold, image):  # 确定好最佳阈值后，将原来的图转变成二值化图
    temp = np.asarray(image)
    print("灰度值矩阵为：")
    print(temp)  # 展现灰度值矩阵
    array = list(np.where(temp < threshold, 0,
                          255).reshape(-1))  # 满足temp<yu，输出0，否则输出255
    image.putdata(array)
    image.show()
    image.save('finished.jpg')

def main():
    image = cv2.imread(photo)
    cv2.imshow('Image', image)  # 先展现出原图
    cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
    ga = GA(gray, 16)
    print("种群变化为：")
    for x in range(100):  # 假设迭代次数为100
        ga.evolution()
        print(ga.species)
    max_yuzhi = ga.best_threshold()
    print("最佳阈值为：", max_yuzhi)
    transition(max_yuzhi, gray)

if __name__ == '__main__':
    main()
