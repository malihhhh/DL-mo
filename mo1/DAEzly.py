from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from DAEclass import DAE
from sklearn import metrics
from myUtils import *
import time

groundtruth = np.loadtxt('data/simulations/equal/5/c.txt')
groundtruth = list(np.int_(groundtruth))

omics1 = np.loadtxt('data/simulations/equal/5/o1.txt')
omics1 = np.transpose(omics1)
omics1 = normalize(omics1, axis=0, norm='max')

omics2 = np.loadtxt('data/simulations/equal/5/o2.txt')
omics2 = np.transpose(omics2)
omics2 = normalize(omics2, axis=0, norm='max')

omics3 = np.loadtxt('data/simulations/equal/5/o3.txt')
omics3 = np.transpose(omics3)
omics3 = normalize(omics3, axis=0, norm='max')

omics = np.concatenate((omics1, omics2, omics3), axis=1)

data = omics

#data_noisy=data+noise_factor * np.random.normal(0.0, 1.0, data.shape)

input_dim = data.shape[1]

encoding1_dim = 500
encoding2_dim = 200
middle_dim = 30

dims = [encoding1_dim, encoding2_dim, middle_dim]

noise_factor=0.1

dae = DAE(data, dims,noise_factor)
dae.autoencoder.summary()
dae.train()
encoded_factors = dae.predict(data)

# #clf = KMeans(n_clusters=5)
# clf=k_means(encoded_factors,5)
# #clf.fit(encoded_factors)
# #centers = clf.cluster_centers_
# #labels = clf.labels_
# # silhouetteScore = silhouette_score(encoded_factors, labels, metric='euclidean')
# # davies_bouldinScore = davies_bouldin_score(encoded_factors, labels)
# # print("----------------------------------")
# # print(centers)
# # print("----------------------------------")
# # print(silhouetteScore)
# # print("----------------------------------")
# # print(davies_bouldinScore)
#
# cluster_centers = clf[0]  # 聚类中心数组
#
# cluster_labels = clf[1]  # 聚类标签数组
#
# plt.scatter(encoded_factors[:,0], encoded_factors[:,1], c=cluster_labels)  # 绘制样本并按聚类标签标注颜色
#
# # 绘制聚类中心点，标记成五角星样式，以及红色边框
#
# for center in cluster_centers:
#     plt.scatter(center[0], center[1], marker="p", edgecolors="red")
#
# plt.show()  # 显示图#print model

# plotting
# encoded_factors = encoder.predict(data)
np.savetxt("AE_FC_5.txt", encoded_factors)
clf = KMeans(n_clusters=5)
t0 = time.time()
clf.fit(encoded_factors)  # 模型训练
km_batch = time.time() - t0  # 使用kmeans训练数据消耗的时间
print("K-Means算法模型训练消耗时间:%.4fs" % km_batch)

# 效果评估
### 效果评估
score_funcs = [
    metrics.adjusted_rand_score,  # ARI（调整兰德指数）
    metrics.v_measure_score,  # 均一性与完整性的加权平均
    metrics.adjusted_mutual_info_score,  # AMI（调整互信息）
    metrics.mutual_info_score,  # 互信息
]

centers = clf.cluster_centers_
print("xmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmu")
print("centers:")
print(centers)
print("xmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmu")
labels = clf.labels_
print("labels:")
print(labels)
print(labels)
print("xmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmuxmu")
print(groundtruth)
print(labels)
## 2. 迭代对每个评估函数进行评估操作
for score_func in score_funcs:
    t0 = time.time()
    km_scores = score_func(groundtruth, labels)
    print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (score_func.__name__, km_scores, time.time() - t0))
t0 = time.time()
jaccard_score = jaccard_coefficient(groundtruth, labels)
print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (jaccard_coefficient.__name__, jaccard_score, time.time() - t0))
silhouetteScore = silhouette_score(encoded_factors, labels, metric='euclidean')
davies_bouldinScore = davies_bouldin_score(encoded_factors, labels)

print("----------------------------------")
print(centers)
print("----------------------------------")
print(silhouetteScore)
print("----------------------------------")
print(davies_bouldinScore)
