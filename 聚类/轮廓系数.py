import numpy as np
from sklearn.cluster import KMeans
from pylab import *
import codecs
import matplotlib.pyplot as plt
import pandas as pd
from numpy.random import random
from sklearn import preprocessing 
from sklearn import metrics
import operator  

data = []
labels = []

datas = np.loadtxt('cluster.dat')
'''
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(datas)
labels = kmeans_model.labels_
a = metrics.silhouette_score(datas, labels, metric='euclidean')
print(a)
'''
silhouette_all=[]

for k in range(2,9):
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(datas)
    labels = kmeans_model.labels_
    a = metrics.silhouette_score(datas, labels, metric='euclidean')
    silhouette_all.append(a)
    #print(a)
    print('这个是k={}次时的轮廓系数：'.format(k),a)
    

dic={}             #存放所有的互信息的键值对
mi_num=2  
for i in silhouette_all:
    dic['k={}时轮廓系数'.format(mi_num)]='{}'.format(i)
    mi_num=mi_num+1
#print(dic)
rankdata=sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
print(rankdata)