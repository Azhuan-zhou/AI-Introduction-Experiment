from collections import deque
import functools
import math


class Node:
    def __init__(self):
        self.name = None  # 节点城市名称
        self.neighbor = {}  # 后继顶点
        self.neighbor_num = 0
        self.axis = (0, 0)
        self.pre = ''
        self.gn = 0  # 该节点到起始节点的距离
        self.hn = 0
        self.fn = 0


class Graph:
    def __init__(self):
        self.nodes = {}  # 图中的节点，通过字典贮存key:城市名称，value:节点信息（节邻居城市信息，节点城市坐标）


graph = Graph()


def create_graph(file1, file2):
    global graph
    with open(file1, 'r', encoding='UTF-8') as f1:
        for line1 in f1.readlines():
            line1 = line1.split()
            num_neighbor = int(line1[1])  # 相邻城市个数
            # 实例化城市节点
            node = Node()
            node.neighbor_num = num_neighbor
            node.name = line1[0]
            for i in range(0, 2 * num_neighbor, 2):
                node.neighbor[line1[i + 2]] = int(line1[i + 3])
            graph.nodes[node.name] = node
    with open(file2, 'r', encoding='utf8') as f2:
        for line2 in f2.readlines():
            line2 = line2.split()
            name = line2[0]
            graph.nodes[name].axis = int(line2[1]), int(line2[2])


def show_graph():
    city_names = graph.nodes.keys()
    for name in city_names:
        print(name,
              'has {} cities in neighbor which are {} and the axis of {} is   {}'
              .format(len(graph.nodes[name].neigbor_num),
                      graph.nodes[name].neighbor,

                      name,
                      graph.nodes[name].axis))


def minimun(close_list, start, end):
    route = [end]
    cost = 0
    pre = graph.nodes[end].pre  # 终点城市的上继城市
    city = end
    while pre != start:
        cost += graph.nodes[pre].neighbor[city]
        route.append(pre)
        city = pre
        pre = graph.nodes[pre].pre
    cost += graph.nodes[pre].neighbor[city]
    route.append(pre)
    route.reverse()
    return cost, route


def show_route(route, cost, end):
    print("The minimun cost is {}".format(cost))
    print("Route:", )
    for i in route:
        if i != end:
            print(i + '<--', end=" ")
        else:
            print(end)


def bfs(start: str, end: str):
    """
    广度优先
    :param start: 起始城市
    :param end: 结束城市
    """
    open_list = deque()  # 存放带扩展的节点
    close_list = []  # 存放已经扩展的节点
    open_list.append(start)  # 讲起始节点放入
    mincost = 10000000
    minroute = []
    while open_list:
        city = open_list.popleft()
        if city not in close_list or city == end:
            if city == end:
                close_list.append(city)
                cost, route = minimun(close_list, start, end)
                if cost < mincost:
                    mincost = cost
                    minroute = route
            else:
                close_list.append(city)
                neighbor = graph.nodes[city].neighbor
                for i in neighbor.keys():
                    if i not in close_list:
                        graph.nodes[i].pre = city
                    open_list.append(i)
    print("-------------------------BFS--------------------------------")
    print("The number of nodes is {}".format(len(close_list)))
    show_route(minroute, mincost, end)
    print("-------------------------------------------------------------")


def dfs(start, end):
    open_list = []
    close_list = []
    mincost = 100000000
    minroute = []
    open_list.append(start)
    while open_list:
        city = open_list.pop()
        if city not in close_list or city == end:
            if city == end:
                close_list.append(city)
                cost, route = minimun(close_list, start, end)
                if cost < mincost:
                    mincost = cost
                    minroute = route
            else:
                close_list.append(city)
                neighbor = graph.nodes[city].neighbor
                for i in neighbor.keys():
                    if i not in close_list:
                        graph.nodes[i].pre = city
                    open_list.append(i)
    print("-------------------------DFS--------------------------------")
    print("The number of nodes is {}".format(len(close_list)))
    show_route(minroute, mincost, end)
    print("-------------------------------------------------------------")


def compute_des(end):  # 计算当前各点到goal的h(n)(欧式距离)
    for i in graph.nodes.keys():
        if i == end:
            graph.nodes[i].hn = 0
        else:
            graph.nodes[i].hn = math.sqrt((int(graph.nodes[i].axis[0]) - int(graph.nodes[end].axis[0])) *
                                          (int(graph.nodes[i].axis[0]) - int(graph.nodes[end].axis[0])) +
                                          (int(graph.nodes[i].axis[1]) - int(graph.nodes[end].axis[1])) *
                                          (int(graph.nodes[i].axis[1]) - graph.nodes[end].axis[1]))


def sot(a, b):
    C1 = graph.nodes[a]
    C2 = graph.nodes[b]
    if C1.fn < C2.fn:  # 总代价小的排前面
        return -1
    if C1.fn > C2.fn:
        return 1
    if C1.fn == C2.fn:  # 总代价相等时，h(n)小的排前面
        if C1.hn < C2.hn:
            return -1
    return 0


def Astar(start, end):
    open_list = deque()  # 待拓展的城市，存的是名字
    close_list = []  # 已拓展的城市，存的是名字
    open_list.append(start)
    compute_des(end)  # 调用函数求h(n)
    while open_list:
        sorted(open_list, key=functools.cmp_to_key(sot))
        print("当前open表的状态是:", open_list, end="")
        print("     当前close表的状态是:", close_list)
        city = open_list.popleft()
        if city not in close_list:
            if (city == end):  # 找到目标城市
                cost, route = minimun(close_list, start, end)
                print('------------------Astar------------------------')
                show_route(route, cost, end)
                print('-------------------------------------------------')
            else:
                close_list.append(city)
                for i in graph.nodes[city].neighbor.keys():
                    if graph.nodes[i].gn == 0:
                        graph.nodes[i].gn = graph.nodes[city].gn + graph.nodes[city].neighbor[i]
                        distance = graph.nodes[i].gn
                    else:
                        distance = graph.nodes[city].gn + graph.nodes[city].neighbor[i]
                    if i in open_list:
                        if graph.nodes[i].gn > distance:
                            graph.nodes[i].pre = city
                    elif i in close_list:
                        if graph.nodes[i].gn > distance:
                            graph.nodes[i].pre = city
                    else:
                        open_list.append(i)
                    graph.nodes[i].fn = graph.nodes[i].gn + graph.nodes[i].hn
                    name = i
                    fn =  graph.nodes[i].fn


if __name__ == '__main__':
    create_graph('cityinfo.txt', 'cityposi.txt')
    bfs('Arad', 'Bucharest')
    dfs('Arad', 'Bucharest')
    Astar('Arad', 'Bucharest')
