import numpy as np
from PIL import Image


class GA:  # 定义遗传算法类
    def __init__(self, image, M):  # 构造函数,进行初始化以及编码
        self.image = image
        self.M = M  # 初始化种群的个体数
        self.length = 8  # 每条染色体基因长度为8(0-255)
        self.species = np.random.randint(0, 256, self.M)  # 给种群随机编码
        self.select_rate = 0.5  # 选择的概率(用于适应性没那么强的染色体)，小于该概率则被选，否则被丢弃
        self.strong_rate = 0.3  # 选择适应性强染色体的比率
        self.bianyi_rate = 0.05  # 变异的概率

    def Adaptation(self, ranseti):  # 进行染色体适应度的评估
        fit = OTSU().otsu(self.image, ranseti)
        return fit

    def selection(self):  # 进行个体的选择，前百分之30的个体直接留下，后百分之70按概率选择
        fitness = []
        for ranseti in self.species:  # 循环遍历种群的每一条染色体，计算保存该条染色体的适应度
            fitness.append((self.Adaptation(ranseti), ranseti))
        fitness1 = sorted(fitness, reverse=True)  # 逆序排序，适应度高的染色体排前面
        for i, j in zip(fitness1, range(self.M)):
            fitness[j] = i[1]
        parents = fitness[:int(len(fitness) *
                               self.strong_rate)]  # 适应性特别强的直接留下来
        for ranseti in fitness[int(len(fitness) *
                                   self.strong_rate):]:  # 挑选适应性没那么强的染色体
            if np.random.random() < self.select_rate:  # 随机比率，小于则留下
                parents.append(ranseti)
        return parents

    def crossover(self, parents):  # 进行个体的交叉
        children = []
        child_count = len(self.species) - len(parents)  # 需要产生新个体的数量
        while len(children) < child_count:
            fu = np.random.randint(0, len(parents))  # 随机选择父亲
            mu = np.random.randint(0, len(parents))  # 随机选择母亲
            if fu != mu:
                position = np.random.randint(0,
                                             self.length)  # 随机选取交叉的基因位置(从右向左)
                mask = 0
                for i in range(position):  # 位运算
                    mask = mask | (1 << i)  # mask的二进制串最终为position个1
                fu = parents[fu]
                mu = parents[mu]
                child = (fu & mask) | (
                    mu & ~mask)  # 孩子获得父亲在交叉点右边的基因、母亲在交叉点左边（包括交叉点）的基因，不是得到两个新孩子
                children.append(child)
        self.species = parents + children  # 产生的新的种群

    def bianyi(self):  # 进行个体的变异
        for i in range(len(self.species)):
            if np.random.random() < self.bianyi_rate:  # 小于该概率则进行变异，否则保持原状
                j = np.random.randint(0, self.length)  # 随机选取变异基因位置点
                self.species[i] = self.species[i] ^ (1 << j)  # 在j位置取反

    def evolution(self):  # 进行个体的进化
        parents = self.selection()
        self.crossover(parents)
        self.bianyi()

    def yuzhi(self):  # 返回适应度最高的这条染色体,为最佳阈值
        fitness = []
        for ranseti in self.species:  # 循环遍历种群的每一条染色体，计算保存该条染色体的适应度
            fitness.append((self.Adaptation(ranseti), ranseti))
        fitness1 = sorted(fitness, reverse=True)  # 逆序排序，适应度高的染色体排前面
        for i, j in zip(fitness1, range(self.M)):
            fitness[j] = i[1]
        return fitness[0]


class OTSU:  # 定义大津算法类
    def otsu(self, image, yuzhi):  # 计算该条染色体(个体)的适应度
        image = np.transpose(np.asarray(image))  # 转置
        size = image.shape[0] * image.shape[1]
        bin_image = image < yuzhi
        summ = np.sum(image)
        w0 = np.sum(bin_image)
        sum0 = np.sum(bin_image * image)
        w1 = size - w0
        if w1 == 0:
            return 0
        sum1 = summ - sum0
        mean0 = sum0 / (w0 * 1.0)
        mean1 = sum1 / (w1 * 1.0)
        fitt = w0 / (size * 1.0) * w1 / (size * 1.0) * (
            mean0 - mean1) * (mean0 - mean1)
        return fitt


def transition(yu, image):  # 确定好最佳阈值后，将原来的图转变成二值化图
    temp = np.asarray(image)
    print("灰度值矩阵为：")
    print(temp)  # 展现灰度值矩阵
    array = list(np.where(temp < yu, 0,
                          255).reshape(-1))  # 满足temp<yu，输出0，否则输出255
    image.putdata(array)
    image.show()
    image.save('D:/2.jpg')


def main():
    tu = Image.open(file)
    tu.show()  # 先展现出原图
    gray = tu.convert('L')  # 转换为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
    ga = GA(gray, 16)
    print("种群变化为：")
    for x in range(100):  # 假设迭代次数为100
        ga.evolution()
        print(ga.species)
    max_yuzhi = ga.yuzhi()
    print("最佳阈值为：", max_yuzhi)
    transition(max_yuzhi, gray)


file = 'test.jpg'
if __name__ == '__main__':
    main()


