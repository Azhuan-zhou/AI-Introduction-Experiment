from math import log
import pandas as pd
import numpy as np
import operator
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import copy

data = pd.read_csv('penguin.csv')  # 从文件中读取数据


def getNumLeafs(myTree):  # 获取决策树叶子结点数目
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if (type(secondDict[key]) is dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):  # 获取决策树层数
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if (type(secondDict[key]) is dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if (thisDepth > maxDepth):
            maxDepth = thisDepth
    return maxDepth


def plotNode(
        nodeTxt, centerPt, parentPt,
        nodeType):  # nodeTxt为结点名centerPt为文本位置parentPt为标注的箭头位置nodeType为结点格式
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    createPlot.ax1.annotate(
        nodeTxt,
        xy=parentPt,
        xycoords='axes fraction',  # 绘制结点
        xytext=centerPt,
        textcoords='axes fraction',
        va="center",
        ha="center",
        bbox=nodeType,
        arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt,
                txtString):  # 标注有向边属性值,cntrPt、parentPt用于计算标注位置,txtString为标注的内容
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,
                        yMid,
                        txtString,
                        va="center",
                        ha="center",
                        rotation=30)


def plotTree(myTree, parentPt, nodeTxt):  # parentPt为标注的内容,nodeTxt为结点名
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt,
                     leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def transition(x):
    """
    将文字类型的信息转换为数值类型
    Species：0为Adelie Penguin (Pygoscelis adeliae) 1为Gentoo penguin 2为Chinstrap penguin
    Island：0为Torgersen  1为Biscoe  2为Dream
    Sex中0为MALE  1为FEMALE 2为NAN
    """
    global data
    if x == data['Species'].unique()[0]:
        return 0
    if x == data['Species'].unique()[1]:
        return 1
    if x == data['Species'].unique()[2]:
        return 2
    if x == data['Island'].unique()[0]:
        return 0
    if x == data['Island'].unique()[1]:
        return 1
    if x == data['Island'].unique()[2]:
        return 2
    if x == data['Sex'].unique()[0]:
        return 0
    if x == data['Sex'].unique()[1]:
        return 1
    if x == data['Sex'].unique()[2]:
        return -1.0


# 创建数据集
def createDataSet():
    global data
    data = data[[
        'Island',  # 岛屿
        'Culmen Length (mm)',  # 嘴巴长度
        'Culmen Depth (mm)',  # 嘴巴深度
        'Flipper Length (mm)',  # 脚蹼长度
        'Body Mass (g)',  # 身体体积
        'Sex',  # 性别
        'Age',  # 年龄
        'Species',  # 种类
    ]]
    data = data.fillna(-1)  # 用-1补充缺失值
    data['Species'] = data['Species'].apply(transition)  # 将文字类型的信息转换为数值类型
    data['Island'] = data['Island'].apply(transition)
    data['Sex'] = data['Sex'].apply(transition)
    dataSet = []  # 存储所有企鹅信息
    for i in range(344):
        dataSet.append(list(data.iloc[i, :]))  # 按每行每行的存
    labels = [
        'Island', 'Culmen Length (mm)', 'Culmen Depth (mm)',
        'Flipper Length (mm)', 'Body Mass (g)', 'Sex', 'Age'
    ]  # 分类特征标签
    return dataSet, labels  # 返回数据集和分类特征


# 计算数据集根节点的信息熵
def calculateEnt(dataSet):
    numPenguis = len(dataSet)  # 企鹅个数
    labelCounts = {}  # 保存每个标签出现的次数
    for featureVector in dataSet:  # 对每组特征向量进行统计(344哥)
        currentLabel = featureVector[-1]  # 通过下标索引找到改组特征向量的标签
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    inforEnt = 0.0  # 初始化信息熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numPenguis  # 选择该标签的概率
        inforEnt -= prob * log(prob, 2)
    return inforEnt  # 返回该数据集的信息熵


def splitDataSet_discrete(dataSet, feature, value):
    # 划分数据集中每个特征的不同值,feature为数据集要划分的特征,value为划分的特征的值
    subDataSet = []
    for featureVector in dataSet:
        if featureVector[feature] == value:
            reduceVector = featureVector[:feature]
            reduceVector.extend(featureVector[feature+1:])
            subDataSet.append(reduceVector)
    return subDataSet  # 返回划分后只留下为第feature特征为value的值的该条数据)的数据集


def splitDataSet_continuous(dataSet, feature, value, method='L'):
    """
     LorR: 取得value值左侧（小于）或右侧（大于）的数据集
    :param dataSet: 企鹅的数据集
    :param feature: 某个特征
    :param value: 某个特征值
    :param method:
    :return:
    """
    subDataSet = []
    if method == 'L':
        for featureVector in dataSet:
            if float(featureVector[feature]) < value:
                reduceVector = featureVector[:feature]
                reduceVector.extend(featureVector[feature + 1:])
                subDataSet.append(reduceVector)
    else:
        for featureVector in dataSet:
            if float(featureVector[feature]) > value:
                reduceVector = featureVector[:feature]
                reduceVector.extend(featureVector[feature + 1:])
                subDataSet.append(reduceVector)
    return subDataSet


# 选择最优特征
def chooseBestFeatureToSplit(dataSet, labelProperty):
    """
    在剩余的特征中寻找信息增益最大的一个特征
    :param dataSet: 企鹅的数据集，包含七个特征和企鹅种类
    :param labelProperty: 特征的离散和连续的区分数组
    :return: 信息增益最大的一个特征
    """
    numFeatures = len(labelProperty)  # 特征长度（7个）
    baseEntropy = calculateEnt(dataSet)  # 计算数据集根节点的信息熵
    bestInfoGain = 0.0  # 最优信息增益
    bestFeature = -1  # 最优特征的索引值
    bestPartValue = None  # 连续特征值中最佳的划分值
    for i in range(numFeatures):
        # 分别把每个特征的值提取出来
        featList = [example[i] for example in dataSet]
        # 不包含重复值的特征值集合
        uniqueValues = set(featList)
        newEntropy = 0.0
        bestPartValuei = None
        # 对离散的特征
        if labelProperty[i] == 0:
            for value in uniqueValues:  # 计算信息增益
                subDataSet = splitDataSet_discrete(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))  # 某个特征值的概率
                newEntropy += prob * calculateEnt(subDataSet)
        # 对连续的特征
        else:
            sortedUniqueValues = list(uniqueValues)
            sortedUniqueValues.sort()  # 升序
            minEntropy = float('inf')  # 选出最小熵时作为划分连续值的节点
            for j in range(len(sortedUniqueValues) - 1):  # 计算划分点
                partValue = (float(sortedUniqueValues[j]) +
                             float(sortedUniqueValues[j + 1])) / 2
                # 对每个划分点，计算信息熵
                dataSetLeft = splitDataSet_continuous(dataSet, i, partValue, 'L')
                dataSetRight = splitDataSet_continuous(dataSet, i, partValue, 'R')
                probLeft = len(dataSetLeft) / float(len(dataSet))
                probRight = len(dataSetRight) / float(len(dataSet))
                Entropy = probLeft * calculateEnt(
                    dataSetLeft) + probRight * calculateEnt(dataSetRight)
                if (Entropy < minEntropy):  # 取最小的信息熵
                    minEntropy = Entropy
                    bestPartValuei = partValue
            newEntropy = minEntropy
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        # 取最大的信息增益对应的特征
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
            bestPartValue = bestPartValuei
    return bestFeature, bestPartValue


def majorityCnt(classList):  # 统计出现次数最多的类的标签
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(),  # 将字典转化为元组
                              key=operator.itemgetter(1),
                              reverse=True)  # 降序
    return sortedClassCount[0][0]


# 后剪枝：用其叶节点代替某些子树(该叶子节点所标识的类别通过大多数原则确定(大多数的类别表示这个叶节点))
# 创建树, 样本集 特征 特征属性（0 离散， 1 连续）
def createTree(dataSet, labels, labelProperty):
    classList = [example[-1] for example in dataSet]  # 返回一个类别中最后一个特征种类的值
    if classList.count(classList[0]) == len(classList):  # 如果只有一个类别，返回
        return classList[0]
    if len(labelProperty) == 1:  # 如果所有特征都被遍历完了，返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeature, bestPartValue = chooseBestFeatureToSplit(dataSet, labelProperty)  # 选择最优分类特征
    # bestFeature是最优特征，bestPartValue是连续型数据的划分点
    if bestFeature == -1:  # 如果无法选出最优分类特征，返回出现次数最多的类别
        return majorityCnt(classList)
    if labelProperty[bestFeature] == 0:  # 对离散的特征
        bestFeatureLabel = labels[bestFeature]  # 离散数据的节点标签
        myTree = {bestFeatureLabel: {}}  # 递归树
        labels_New = copy.copy(labels)  # 浅拷贝
        labelProperty_New = copy.copy(labelProperty)
        # 将已经选择过的特征删去，不再参与分类
        del (labels_New[bestFeature])
        del (labelProperty_New[bestFeature])
        featureValues = [example[bestFeature] for example in dataSet]  # 344个企鹅中最好特征的特征值
        uniqueValue = set(featureValues)  # 该特征包含的所有值
        for value in uniqueValue:  # 对每个特征值，递归构建树
            subLabels = labels_New[:]
            subLabelProperty = labelProperty_New[:]
            myTree[bestFeatureLabel][value] = createTree(
                splitDataSet_discrete(dataSet, bestFeature, value), subLabels,
                subLabelProperty)
    else:  # 对连续的特征分别构建左子树和右子树
        bestFeatureLabel = labels[bestFeature] + '<' + str(bestPartValue)  # 连续数据的节点标签
        myTree = {bestFeatureLabel: {}}
        labels_New = copy.copy(labels)  # 浅拷贝
        labelProperty_New = copy.copy(labelProperty)
        del(labels_New[bestFeature])
        del[labelProperty_New[bestFeature]]
        subLabels = labels_New[:]
        subLabelProperty = labelProperty_New[:]
        # 构建左子树
        valueLeft = 'Yes'
        myTree[bestFeatureLabel][valueLeft] = createTree(
            splitDataSet_continuous(dataSet, bestFeature, bestPartValue, 'L'), subLabels,
            subLabelProperty)
        # 构建右子树
        valueRight = 'No'
        myTree[bestFeatureLabel][valueRight] = createTree(
            splitDataSet_continuous(dataSet, bestFeature, bestPartValue, 'R'), subLabels,
            subLabelProperty)
    return myTree


def classify(inputTree, featLabels, featLabelProperties, testVec):
    firstStr = list(inputTree.keys())[0]  # 根节点
    firstLabel = firstStr
    lessIndex = str(firstStr).find('<')
    if lessIndex != -1:  # 如果是连续型的特征
        firstLabel = str(firstStr)[:lessIndex]
    secondDict = inputTree[firstStr]  # 第二个字典
    featIndex = featLabels.index(firstLabel)  # 根节点对应的特征的索引
    classLabel = None  # 分类标签
    for key in secondDict.keys():  # 对每个分支循环
        if featLabelProperties[featIndex] == 0:  # 离散的特征
            if testVec[featIndex] == key:  # 判断测试的向量中，该特征值所对应的下一分支
                if type(secondDict[key]).__name__ == 'dict':  # 判断是否到达叶子节点
                    classLabel = classify(secondDict[key], featLabels,
                                          featLabelProperties, testVec)
                else:  # 若到达叶子节点返回
                    classLabel = secondDict[key]
        else:  # 连续的特征
            partValue = float(str(firstStr)[lessIndex + 1:])  # 划分点值
            if testVec[featIndex] < partValue:  # 进入左子树
                if type(secondDict['Yes']).__name__ == 'dict':  # 判断是否到达叶子节点
                    classLabel = classify(secondDict['Yes'], featLabels,
                                          featLabelProperties, testVec)
                else:  # 若到达叶子节点返回
                    classLabel = secondDict['Yes']
            else:
                if type(secondDict['No']).__name__ == 'dict':  # 判断是否到达叶子节点
                    classLabel = classify(secondDict['No'], featLabels,
                                          featLabelProperties, testVec)
                else:  # 若到达叶子节点返回
                    classLabel = secondDict['No']

    return classLabel


def main():
    global data
    dataSet, labels = createDataSet()
    labelProperties = [0, 1, 1, 1, 1, 0, 1]  # 属性的类型，0表示离散，1表示连续
    myTree = createTree(dataSet, labels, labelProperties)
    feature = data[[
        'Island', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)',
        'Body Mass (g)', 'Sex', 'Age'
    ]]  # 企鹅的七种特征
    goal = data['Species']
    x_train, x_test, y_train, y_test = train_test_split(feature,
                                                        goal,
                                                        test_size=0.2)
    x_test = np.array(x_test)
    x_test = x_test.tolist()
    test_pre = []  # 存放预测结果的类别标签
    for i in range(len(x_test)):
        result = classify(myTree, labels, labelProperties, x_test[i])
        test_pre.append(result)
    print("预测准确率为:", accuracy_score(y_test, test_pre))
    confu_matrix = confusion_matrix(test_pre, y_test)  # 查看混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(confu_matrix, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()
    createPlot(myTree)  # 决策树可视化
    print(myTree)


if __name__ == '__main__':
    main()
