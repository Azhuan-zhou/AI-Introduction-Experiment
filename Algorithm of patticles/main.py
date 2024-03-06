import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

customers = []  # 顾客信息，（坐标+客户需求）


def read_data(file):
    """
    从文件中读取客户信息
    """
    f = open(file, 'r', encoding='UTF-8')
    data = f.readlines()
    f.close()
    global customers
    for i in range(len(data)):
        customers.append(eval(data[i]))


def calculation_distance():
    """
    计算城市间距离
    输入：city_coordinates-城市坐标；
    输出：城市间距离矩阵-dis_matrix
    """
    # 从客户信息中获取客户的坐标
    city_coordinates = []
    for i in range(len(customers)):
        city_coordinates.append(customers[i][0])
    distance_matrix = pd.DataFrame(data=None, columns=range(len(customers)), index=range(len(customers)))  # 城市间距离矩阵
    for i in range(len(city_coordinates)):
        xi, yi = city_coordinates[i][0], city_coordinates[i][1]
        for j in range(len(city_coordinates)):
            xj, yj = city_coordinates[j][0], city_coordinates[j][1]
            distance_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)
    return distance_matrix


def greedy(distance_matrix):
    # 修改dis_matrix以适应求解需要
    distance_matrix = distance_matrix.astype('float64')
    for i in range(len(customers)):
        distance_matrix.loc[i, i] = float('inf')  # 设置点i到点i的距离为无穷大
    distance_matrix.loc[:, 0] = float('inf')  # 设置点i到起始点的距离为无穷大
    now_customer = np.random.randint(1, len(customers))
    route = [now_customer]
    distance_matrix.loc[:, now_customer] = float('inf')  # 更新距离矩阵，已经过城市不再被取出
    for i in range(1, len(customers)-1):
        next_costumer = distance_matrix.loc[now_customer, :].idxmin()  # 最近的城市
        route.append(next_costumer)  # 添加进路径
        distance_matrix.loc[:, next_costumer] = float('inf')  # 更新距离矩阵
        now_customer = next_costumer  # 更新当前城市
    return route


def calculation_fitness(routes, distance_matrix, capacity, distance, c0, c1):
    """
    在使用贪婪策略分配完车辆后，对路线进行解码，分配车辆，计算路径距离
    :param routes: 路径
    :param demand: 客户需求
    :param distance_matrix: 城市间距离矩阵
    :param capacity: 车辆最大载重
    :param distance:车辆最大行驶距离
    :param c0:车辆启动成本
    :param c1:车辆单位距离行驶成本
    :return:routes_car-分车后路径,fitness-适应度
    """
    routes_car, fitness = [], []
    for j in range(len(routes)):
        route = routes[j]  # 当前路径
        lines = []  # 存储线路分车
        line = [0]  # 每辆车服务客户点
        dis_sum = 0  # 线路距离
        dis, d = 0, 0  # 当前客户距离前一个客户的距离、当前客户需求量
        i = 0  # 指向配送中心
        while i < len(route):
            if line == [0]:  # 车辆未分配客户点
                dis += distance_matrix.loc[0, route[i]]  # 记录距离
                d += customers[route[i]][1]  # 记录需求量
                line.append(route[i])  # 为客户点分车
                i += 1  # 指向下一个客户点
            else:  # 已分配客户点则需判断车辆载重和行驶距离
                if (distance_matrix.loc[line[-1], route[i]] + distance_matrix.loc[route[i], 0] + dis <= distance) & (
                        d + customers[route[i]][1] <= capacity):  # 当前距离加上下一个点的距离和下一个点与起始点的距离小于一辆车行驶的上限距离，并且与下个点的需求和小于车辆运载上限
                    dis += distance_matrix.loc[line[-1], route[i]]
                    line.append(route[i])
                    d += customers[route[i]][1]
                    i += 1
                else:
                    dis += distance_matrix.loc[line[-1], 0]  # 当前车辆装满
                    line.append(0)
                    dis_sum += dis
                    lines.append(line)
                    # 下一辆车
                    dis, d = 0, 0
                    line = [0]

        # 最后一辆车
        dis += distance_matrix.loc[line[-1], 0]
        line.append(0)
        dis_sum += dis
        lines.append(line)

        routes_car.append(lines)
        fitness.append(round(c1 * dis_sum + c0 * len(lines), 1))

    return routes_car, fitness

def crossover(parent, present_route, global_route, w, c1, c2):
    """
    采用顺序交叉方式；交叉的parent1为粒子本身，分别以w/(w+c1+c2),c1/(w+c1+c2),c2/(w+c1+c2)
    的概率接受粒子本身逆序、当前最优解、全局最优解作为parent2,只选择其中一个作为parent2；
    :param parent: 粒子
    :param present_line: 当前最优解
    :param global_line: 全局最优解
    :param w: 惯性因子
    :param c1: 自我认知因子
    :param c2: 社会认知因子
    :return: 交叉后的粒子
    """
    children = [None]*len(parent)
    # 上一代1
    parent1 = parent
    # 上一代2
    k = np.random.uniform(0, sum([w, c1, c2]))
    if k <= w:
        # 逆序
        parent2 = [parent[i] for i in range(len(parent)-1, -1, -1)]
    elif k <= w+c1:
        # 当前最优解
        parent2 = present_route
    else:
        parent2 = global_route
    # 交叉parent1 和 parent2
    # parent1
    start = np.random.randint(0, len(parent))
    end = np.random.randint(0, len(parent))
    if start > end:
        start, end = end, start
    children[start:end+1] = parent1[start:end+1].copy()
    # parent2
    list2 = list(range(0, start))
    list1 = list(range(end + 1, len(parent2)))
    list_index = list1 + list2  # croBird从后往前填充
    for i in list_index:
        for j in range(0, len(parent2)):
            if parent2[j] not in children:
                children[i] = parent2[j]
                break
    return children

def draw_path(car_routes):
    """
    画路径图
    :param car_routes: 车辆路径
    :return: 路径图
    """
    for route in car_routes:
        x, y = [], []
        for i in route:
            coordinate = customers[i][0]
            x.append(coordinate[0])
            y.append(coordinate[1])
        x.append(x[0])
        y.append(y[0])
        plt.plot(x, y, 'o-', alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    # 车辆参数
    capacity = 120  # 车辆最大容量
    distance = 250  # 车辆最大行驶距离
    C0 = 30  # 车辆启动成本
    C1 = 1  # 车辆单位距离行驶成本
    # POS参数
    particle_num = 50  # 粒子数量
    w = 0.2  # 惯性因子
    c1 = 0.4  # 自我认知因子
    c2 = 0.4  # 社会认知因子
    # 迭代的参数
    iterNum = 800  # 迭代次数
    iterI = 1  # 当前迭代次数
    bestfitness = []  # 记录每次最优值
    # 读取数据
    read_data('data.txt')
    # 计算城市间的距离
    distance_matrix = calculation_distance()
    # 利用贪心法初始化路线，一共有50个粒子
    routes = [greedy(distance_matrix) for i in range(particle_num)]
    # 为每条路线分配车辆
    routes_car, fitness = calculation_fitness(routes, distance_matrix, capacity, distance, C0, C1)
    # 全局最优值，当前最优值
    global_best = present_best = min(fitness)
    # 全局最优解、当前最优解
    global_route = present_route = routes[fitness.index(global_best)]
    # 全局最优解的车辆分配，当前最优解的车辆分配
    global_route_car = present_route_car = routes_car[fitness.index(global_best)]
    bestfitness.append(global_best)
    # 开始迭代
    while iterI <= iterNum:
        for i in range(particle_num):  # 通过遗传交叉更新粒子的路径
            routes[i] = crossover(routes[i], present_route,global_route,w,c1,c2)
        routes_car, fitness = calculation_fitness(routes, distance_matrix,capacity,distance,C0,C1)
        # 计算当前最优值，当前最优解，当前最优解的车辆分配
        present_best,present_route,present_route_car = min(fitness), routes[fitness.index(min(fitness))],routes_car[fitness.index(min(fitness))]
        # 计算全局最优值,全局最优解,全局最优解的车辆分配
        if min(fitness) < global_best:
            global_best, global_route, global_route_car = present_best,present_route,present_route_car
        bestfitness.append(global_best)
        print('{}:{}'.format(iterI, global_best))
        iterI += 1
    j = 1
    for i in global_route_car:
        print("第{}辆车的路线：{}".format(j, i), end='\n')
        j += 1
    draw_path(global_route_car)
