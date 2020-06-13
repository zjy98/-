# -*- coding: utf-8 -*-
import operator
from math import log
 
#统计classList中出现此处最多的元素（类标签）,即选择出现次数最多的结果
def majorityCnt(classList):
    classCount={}
    for vote in classList:
    #统计classList中每个元素出现的次数
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    #返回classList中出现次数最多的元素
 
#程序清单3-4：创建树
def createTree(dataSet, labels):
    #创建一个列表，包含所有的类标签（数据集的最后一列是标签）
    classList = [example[-1] for example in dataSet]
    #所有的类标签完全相同，则直接返回该类标签
    #列表中第一个值（标签）出现的次数==整个集合的数量，也就是说只有一个类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #使用完了所有特征，仍然不能将数据集划分为仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  #挑选出现次数最多的类别作为返回值
 
    # 选择最优的列，得到最优列对应的label的含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #获取label的名称
    bestFeatLabel = labels[bestFeat]
    #初始化mytree
    mytree = {bestFeatLabel:{}}
    #在标签列表中删除当前最优的标签
    del(labels[bestFeat])
    #得到最优特征包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    #去除重复的特征值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #求出剩余的标签label
        subLabels = labels[:]
        #遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用createTree()函数
        mytree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat,value), subLabels)
    return mytree
 
#程序清单3-3：选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 #求第一行有多少列的特征feature，label在最后一列      
    baseEntropy = calcShannonEnt(dataSet) #计算整个数列集的原始香农熵
    bestInfoGain = 0.0 #最优的信息增益值
    bestFeature = -1 #最优的特征索引值    
    for i in range(numFeatures):  #循环遍历所有特征
        #使用(List Comprehension)来创建新的列表
        #获取数据集中所有的第i个特征值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) #创建set集合{},set类型中每个值互不相同
        newEntropy = 0.0 #创建一个临时的信息熵
        # 经验条件熵
        # 遍历某一列的value集合，计算该列的信息熵
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和
        for value in uniqueVals: #计算信息增益
            subDataSet = splitDataSet(dataSet, i, value) #subDataSet划分后的子集
            prob = len(subDataSet)/float(len(dataSet)) #计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet) #根据公式计算经验条件熵     
        #信息增益：划分数据集前后的信息变化，获取信息熵最大的值
        infoGain = baseEntropy - newEntropy 
        #比较所有特征中的信息增益，返回最好特征划分的索引值            
        if (infoGain > bestInfoGain): #计算信息增益        
            bestInfoGain = infoGain #更新信息增益，找到最大的信息增益         
            bestFeature = i #记录信息增益最大的特征的索引值            
    return bestFeature #返回信息增益最大的特征的索引值
 
    # 程序清单3-1：计算给定数据集的香农熵(经验熵)
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #返回数据集的行数
    labelCounts = {} #保存每个标签(Label)出现次数的字典    
    for featVec in dataSet: #对每组特征向量进行统计        
        currentLabel = featVec[-1] #提取标签(Label)信息，每一行的最后一个数据表示的是标签
        #如果标签(Label)没有放入统计次数的字典,添加进去
        if currentLabel not in labelCounts.keys():         
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 #Label计数
    shannonEnt = 0.0 #经验熵(香农熵)
    for key in labelCounts: #计算香农熵    
        prob = float(labelCounts[key])/numEntries #选择该标签(Label)的概率
        shannonEnt -= prob * log(prob,2) #计算香农熵，以2为底求对数，信息期望值     
    return shannonEnt #返回经验熵(香农熵)
 
#程序清单3-2：按照给定特征划分数据集
def splitDataSet(dataSet, axis, value): # dataSet——待划分的数据集 axis——划分数据集的特征 value——特征的返回值
    retDataSet = []
    #创建新列表存储返回数据列表对象
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            #去掉axis特征     
            reducedFeatVec.extend(featVec[axis+1:])
            #将符合条件的添加到返回的数据里表中
            retDataSet.append(reducedFeatVec)
    return retDataSet
    #返回划分后的数据集
 
#数据集
def createDataSet():
    dataSet = [[1, 1,0,0,1,0,0,1,1,0, '姚明'],
              [1,1,0,0,1,1,0,0,1,0, '刘翔'],
              [1,1,1,1,0,0,0,1,0,0,'科比'],
              [1,1,0,0,1,0,0,0,0,0, 'C罗'],
              [1,0,0,0,0,0,0,0,0,1, '刘德华'],
              [1,0,0,0,0,0,1,0,1,0, '毛不易'],
              [1,0,1,0,0,0,0,0,0,1 ,'周杰伦'],
              [1,0,1,0,0,0,0,0,1,1 ,'黄渤'],
              [1,0,1,1,0,0,0,0,1,1 ,'徐峥'],
              [0,1,0,0,1,0,0,0,1,0 ,'张怡宁'],
              [0,1,0,0,0,1,0,0,1,0 ,'郎平'],
              [0,1,0,0,0,0,0,0,1,0 ,'朱婷'],
              [0,0,0,0,0,0,1,0,1,1 ,'杨超越'],
              [0,0,0,0,1,1,0,0,1,1 ,'杨幂'],
              [0,0,0,0,0,0,0,0,0,0 ,'邓紫棋'],
              [0,0,0,0,1,0,1,0,0,0 ,'徐佳莹'],
              [0,0,0,0,1,0,0,0,1,1 ,'赵丽颖']
              ] #数据集
    labels = ['男','运动员','70后','光头','80后','离婚','选秀','篮球','内地','演员'] #分类属性
    return dataSet, labels #返回数据集和分类属性
 
myDat, labels = createDataSet() #调用数据集
print(myDat,'\n')
A = createTree(myDat, labels)
print(A,'\n')

import matplotlib.pyplot as plt
 
#添加这段代码的目的是让图片中的中文能够正常显示！
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']  
 
#程序清单3-6：获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0 #初始化叶子
    firstStr = list(myTree.keys())[0] #获取结点属性,第一个关键字，第一次划分数据集的类别标签
    secondDict = myTree[firstStr] #获取下一组字典    
    #从根节点开始遍历
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
        #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点        
            numLeafs += getNumLeafs(secondDict[key]) #递归调用
        else:   
            numLeafs +=1
    return numLeafs
 
#程序清单3-6：获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0 #初始化决策树深度
    firstStr = list(myTree.keys())[0] #获取结点属性
    secondDict = myTree[firstStr] #获取下一组字典
    #根节点开始遍历
    for key in secondDict.keys():
        #判断节点的个数，终止条件是叶子节点
        if type(secondDict[key]).__name__=='dict':
        #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   
            thisDepth = 1
        if thisDepth > maxDepth: 
            maxDepth = thisDepth
        #更新层数
    return maxDepth
{'男': {0: {'80后': {0: {'运动员': {0: {'选秀': {0: '邓紫棋', 1: '杨超越'}}, 1: {'离婚': {0: '朱婷', 1: '郎平'}}}}, 1: {'演员': {0: {'运
动员': {0: '徐佳莹', 1: '张怡宁'}}, 1: {'离婚': {0: '赵丽颖', 1: '杨幂'}}}}}}, 1: {'运动员': {0: {'70后': {0: {'选秀': {0: '刘德华', 1: 
'毛不易'}}, 1: {'光头': {0: {'内地': {0: '周杰伦', 1: '黄渤'}}, 1: '徐峥'}}}}, 1: {'篮球': {0: {'离婚': {0: 'C罗', 1: '刘翔'}}, 1: {'70 
后': {0: '姚明', 1: '科比'}}}}}}}}

#测试数据集，输出存储的树信息
def retrieveTree(i):
    listOfTrees =[]
    return listOfTrees[i]
 
def plotMidText(cntrPt, parentPt, txtString):
    #在父子节点间填充文本信息
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    #createPlot.ax1.text(xMid, yMid, txtString)
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation = 30)
 
def plotTree(myTree, parentPt, nodeTxt):
    #计算宽与高
    numLeafs = getNumLeafs(myTree)
    defth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    #找到第一个中心点的位置，然后与parentPt定点进行划线
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)  #中心位置
    #打印输入对应的文字
    plotMidText(cntrPt, parentPt, nodeTxt)
    #可视化Node分支点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondeDict = myTree[firstStr]  #下一个字典
    #减少y的偏移，按比例减少 ，y值 = 最高点 - 层数的高度[第二个节点位置]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondeDict.keys():
        #这些节点既可以是叶子结点也可以是判断节点
        #判断该节点是否是Node节点
        if type(secondeDict[key]) is dict:
            #如果是就递归调用
            plotTree(secondeDict[key], cntrPt, str(key))
        else:
            #如果不是，就在原来节点一半的地方找到节点的坐标
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            #可视化该节点的位置
            plotNode(secondeDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            #并打印输入对应的文字
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
 
#创建绘图区，计算树形图的全局尺寸
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    #清空当前图像窗口
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    #存储树的宽度
    plotTree.totalW = float(getNumLeafs(inTree))
    #存储树的深度
    plotTree.totalD = float(getTreeDepth(inTree))
    #追踪已经绘制的节点位置，以及放置下个节点的恰当位置
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
 
#绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords ='axes fraction', xytext = centerPt,
                            textcoords = 'axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
 
#定义文本框和箭头格式，sawtooth 波浪方框， round4矩形方框， fc表示字体颜色的深浅 0.1~0.9依次变浅
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode = dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")
 
myTree = retrieveTree(0)
createPlot(myTree)