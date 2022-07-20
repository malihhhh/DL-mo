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
from SVAEclass import VAE
import os
from keras import backend as K


def get_EM(datapath,resultpath):
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



    encoding1_dim1 = 1000
    encoding2_dim1 = 100
    middle_dim1 = 4
    dims1 = [encoding1_dim1, encoding2_dim1, middle_dim1]
    ae1 = VAE(omics1, dims1)
    ae1.train()
    ae1.autoencoder.summary()
    encoded_factor1 = ae1.predict(omics1)

    encoding1_dim2 = 500
    encoding2_dim2 = 50
    middle_dim2 = 2
    dims2 = [encoding1_dim2, encoding2_dim2, middle_dim2]
    ae2 = VAE(omics2, dims2)
    ae2.train()
    ae2.autoencoder.summary()
    encoded_factor2 = ae2.predict(omics2)

    encoding1_dim3 = 1000
    encoding2_dim3 = 100
    middle_dim3 = 4
    dims3 = [encoding1_dim3, encoding2_dim3, middle_dim3]
    ae3 = VAE(omics3, dims3)
    ae3.autoencoder.summary()
    ae3.train()
    encoded_factor3 = ae3.predict(omics3)

    encoded_factors = np.concatenate((encoded_factor1, encoded_factor2, encoded_factor3), axis=1)
    if not os.path.exists("{}/SVAE_FAETC_EM.txt".format(resultpath)):
        os.mknod("{}/SVAE_FAETC_EM.txt".format(resultpath))
    np.savetxt("{}/SVAE_FAETC_EM.txt".format(resultpath), encoded_factors)
    K.clear_session()


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
    # data_dir_list=['data/cancer/breast', 'data/cancer/gbm', 'data/cancer/ovarian', 'data/cancer/sarcoma', 'data/cancer/lung', 'data/cancer/liver']
    # result_dir_list=['result/cancer/breast', 'result/cancer/gbm', 'result/cancer/ovarian', 'result/cancer/sarcoma', 'result/cancer/lung', 'result/cancer/liver']

    for datapath,resultpath in zip(data_dir_list,result_dir_list):
        get_EM(datapath, resultpath)
    # datapath='data/cancer/gbm'
    # resultpath='result/cancer/gbm'
    get_EM(datapath, resultpath)









