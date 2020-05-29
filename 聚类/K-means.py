# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import shutil

data = np.loadtxt('cluster.dat')

# 计算数据点到中心点的距离,
def Distance(A1,A2):
    return np.sqrt(sum((A2-A1) ** 2))


# 初始化中心点
# 因为对样本不熟悉，所以采用随机的方法初始化挑选
def initcenterid(data,k):
    num,dim = data.shape
    centerid = np.zeros((k,dim))
    for i in range(k):
        index = int(np.random.uniform(0,num))
        centerid[i,:] = data[index,:]
    return centerid

def Kmeans(data,k):
    num = data.shape[0]
    clusterData = np.array(np.zeros((num,2))) #创建矩阵保存聚类结果和误差的平方[中心点索引,误差]
    clusterChanged = True
    centerid = initcenterid(data,k) #随机初始化K个中心点
    while clusterChanged:
        clusterChanged = False
        for i in range(num): #对每个数据点分配距离最近的中心点
            minDist = 100000.0
            minIndex = 0
            for j in range(k):
                distance = Distance(centerid[j,:],data[i,:]) #计算点到中心点的距离
                if distance < minDist:
                    minDist = distance #如果第i个数据点到第j个中心点更近，则将i归属为j
                    clusterData[i,1] = minDist
                    minIndex = j
            if clusterData[i,0] != minIndex: #如果分配发生变化，则需要继续迭代,一直到不变为止
                clusterChanged = True
                clusterData[i,0] = minIndex
        for j in range(k):    # 重新确定中心点
            cluster_index = np.nonzero(clusterData[:,0] == j)
            pointsInCluster = data[cluster_index] 
            centerid[j,:] = np.mean(pointsInCluster,axis=0) #计算每簇点坐标的均值求新的中心点存入centerid[中心点索引，中心点坐标]
    return centerid,clusterData

# 结果可视化
def showCluster(data, k, centerid, clusterData):
    num, dim = data.shape
    # 用不同颜色形状来表示各个类别
    mark = ['or', 'ob', 'og', 'oc', 'ok', 'om', 'oy', '+b', 'or']
    if k > len(mark):
        print('your k is too large!')
        return 1
    # 画样本点
    for i in range(num):
        markIndex = int(clusterData[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[markIndex])
    # 画中心点
    mark = ['*y', '*y', '*y', '*y', '*y', '*y', '*y', '*y', '*y', '*y']

    for i in range(k):
        plt.plot(centerid[i, 0], centerid[i, 1], mark[i], markersize=20)
    plt.show()


# 将数据集划分为8:2的训练集和测试集
def getData(data):
    data = data.tolist()
    dataNumber = data.__len__()  # 数据集数据条数
    testNumber = int(dataNumber * 0.2)  # 测试集数据条数
    testDataSet = []  # 测试数据集
    trainDataSet = []  # 训练数据集
 
    testDataSet = random.sample(data, testNumber)  # 测试集
    for testData in testDataSet:  # 将已经选定的测试集数据从数据集中删除
        data.remove(testData)
    trainDataSet = data  # 训练集
    l1 = trainDataSet.__len__()
    print(l1)
    l2 = testDataSet.__len__()
    print(l2)
    return trainDataSet, testDataSet

trainData,testData = getData(data)
trainData = np.array(trainData)
testData = np.array(testData)

# 显示聚类结果
k = 4
centerid,clusterData,SSE = Kmeans(data,k)
if np.isnan(centerid).any():
    print('Error')
else:
    print('cluster complete!')
    #显示结果
showCluster(data,k,centerid,clusterData)

# 计算SSE，利用肘部法则评价最优K值
'''
Arr_SSE=[] #存放K值不同时的SSE值
for i in range(2,9):
  centerid,clusterData = Kmeans(trainData,i) #分别用data,trainData,testData进行测试
  loss = sum((clusterData[:,1]) **2)
  Arr_SSE.append(loss)

print(Arr_SSE)
plt.plot(range(2,9),Arr_SSE,marker="o")
plt.xlabel("K")
plt.ylabel("SSE")
plt.show()
'''