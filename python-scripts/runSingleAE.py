from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
import time
from sklearn import metrics
from myUtils import *
from AEclass import AE
import os

if __name__ == '__main__':
    datapath = 'data/single-cell/'
    resultpath = 'result/single-cell/'
    # groundtruth = np.loadtxt('{}/c.txt'.format(datapath))
    # groundtruth = list(np.int_(groundtruth))

    omics = np.loadtxt('{}/omics.txt'.format(datapath))
    omics = np.transpose(omics)
    omics1=omics[0:206]
    omics2=omics[206:412]
    omics1 = normalize(omics1, axis=0, norm='max')
    omics2 = normalize(omics2, axis=0, norm='max')
    omics = np.concatenate((omics1, omics2), axis=1)

    # omics1 = np.loadtxt('{}/ata.txt'.format(datapath))
    # omics1 = np.transpose(omics1)
    # omics1 = normalize(omics1, axis=0, norm='max')
    #
    # omics2 = np.loadtxt('{}/rna.txt'.format(datapath))
    # omics2 = np.transpose(omics2)
    # omics2 = normalize(omics2, axis=0, norm='max')
    #
    # omics = np.concatenate((omics1, omics2), axis=1)

    print(omics1.shape)
    print(omics2.shape)

    # data = omics
    # input_dim = data.shape[1]
    # encoding1_dim = 4096
    # encoding2_dim = 1024
    # middle_dim = 2
    # dims = [encoding1_dim, encoding2_dim, middle_dim]
    # ae = AE(data, dims)
    # ae.autoencoder.summary()
    # ae.train()
    # encoded_factors = ae.predict(data)
    # if not os.path.exists("{}/AE_FCTAE_EM.txt".format(resultpath)):
    #     os.mknod("{}/AE_FCTAE_EM.txt".format(resultpath))
    # np.savetxt("{}/AE_FCTAE_EM.txt".format(resultpath), encoded_factors)




    #
    # # if not os.path.exists("AE_FCTAE_Kmeans.txt"):
    # #     os.mknod("AE_FCTAE_Kmeans.txt")
    # # fo = open("AE_FCTAE_Kmeans.txt", "a")
    # clf = KMeans(n_clusters=typenum)
    # t0 = time.time()
    # clf.fit(encoded_factors)  # 模型训练
    # km_batch = time.time() - t0  # 使用kmeans训练数据消耗的时间
    #
    # print(datatype, typenum)
    # print("K-Means算法模型训练消耗时间:%.4fs" % km_batch)
    #
    # # 效果评估
    # score_funcs = [
    #     metrics.adjusted_rand_score,  # ARI（调整兰德指数）
    #     metrics.v_measure_score,  # 均一性与完整性的加权平均
    #     metrics.adjusted_mutual_info_score,  # AMI（调整互信息）
    #     metrics.mutual_info_score,  # 互信息
    # ]
    # centers = clf.cluster_centers_
    # # print("centers:")
    # # print(centers)
    # print("zlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzly")
    # labels = clf.labels_
    # print("labels:")
    # print(labels)
    # labels = list(np.int_(labels))
    # if not os.path.exists("{}/AE_FCTAE_CL.txt".format(resultpath)):
    #     os.mknod("{}/AE_FCTAE_CL.txt".format(resultpath))
    # np.savetxt("{}/AE_FCTAE_CL.txt".format(resultpath), labels, fmt='%d')
    # print("zlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzly")
    # # 2. 迭代对每个评估函数进行评估操作
    # for score_func in score_funcs:
    #     t0 = time.time()
    #     km_scores = score_func(groundtruth, labels)
    #     print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (score_func.__name__, km_scores, time.time() - t0))
    # t0 = time.time()
    # jaccard_score = jaccard_coefficient(groundtruth, labels)
    # print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (
    #     jaccard_coefficient.__name__, jaccard_score, time.time() - t0))
    # silhouetteScore = silhouette_score(encoded_factors, labels, metric='euclidean')
    # davies_bouldinScore = davies_bouldin_score(encoded_factors, labels)
    # print("silhouetteScore:", silhouetteScore)
    # print("davies_bouldinScore:", davies_bouldinScore)
    # print("zlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzly")







