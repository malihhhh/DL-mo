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
    data_dir_list = []
    result_dir_list = []
    data_path = r"data/cancer"
    result_path = r"result/cancer"
    dir_or_files = os.listdir(data_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        data_dir_file_path = os.path.join(data_path, dir_file)
        result_dir_file_path = os.path.join(result_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(data_dir_file_path):
            data_dir_list.append(data_dir_file_path)
            if not os.path.exists(result_dir_file_path):
                os.makedirs(result_dir_file_path)
            result_dir_list.append(result_dir_file_path)
    #print(data_dir_list)
    #print(result_dir_list)
    #data_dir_list=['data/cancer/breast', 'data/cancer/kidney', 'data/cancer/lung', 'data/cancer/liver']
    #result_dir_list=['result/cancer/breast', 'result/cancer/kidney', 'result/cancer/lung', 'result/cancer/liver']

    for datapath,resultpath in zip(data_dir_list,result_dir_list):
        print(datapath)
        omics1 = np.loadtxt('{}/log_exp_omics.txt'.format(datapath))
        omics1 = np.transpose(omics1)
        omics1 = normalize(omics1, axis=0, norm='max')
        print(omics1.shape)
        omics2 = np.loadtxt('{}/log_mirna_omics.txt'.format(datapath))
        omics2 = np.transpose(omics2)
        omics2 = normalize(omics2, axis=0, norm='max')
        print(omics2.shape)
        omics3 = np.loadtxt('{}/methy_omics.txt'.format(datapath))
        omics3 = np.transpose(omics3)
        omics3 = normalize(omics3, axis=0, norm='max')
        print(omics3.shape)
        omics = np.concatenate((omics1, omics2, omics3), axis=1)
        print(omics.shape)
        print('*'*100)
        # data = omics
        # input_dim = data.shape[1]
        # encoding1_dim = 3000
        # encoding2_dim = 300
        # middle_dim = 10
        # dims = [encoding1_dim, encoding2_dim, middle_dim]
        # ae = AE(data, dims)
        # #ae.autoencoder.summary()
        # ae.train()
        # encoded_factors = ae.predict(data)
        # if not os.path.exists("{}/AE_FCTAE_EM.txt".format(resultpath)):
        #     os.mknod("{}/AE_FCTAE_EM.txt".format(resultpath))
        # np.savetxt("{}/AE_FCTAE_EM.txt".format(resultpath), encoded_factors)






