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
#from AEclass import AE
from SVAEclass import VAE
import os

if __name__ == '__main__':
    datatypes=["equal","heterogeneous"]
    typenums=[5,10,15]
    for datatype in datatypes:
        for typenum in typenums:
            datapath='data/simulations/{}/{}'.format(datatype, typenum)
            resultpath='result/simulations/{}/{}'.format(datatype, typenum)
            groundtruth = np.loadtxt('{}/c.txt'.format(datapath))
            groundtruth = list(np.int_(groundtruth))

            omics1 = np.loadtxt('{}/o1.txt'.format(datapath))
            omics1 = np.transpose(omics1)
            omics1 = normalize(omics1, axis=0, norm='max')

            omics2 = np.loadtxt('{}/o2.txt'.format(datapath))
            omics2 = np.transpose(omics2)
            omics2 = normalize(omics2, axis=0, norm='max')

            omics3 = np.loadtxt('{}/o3.txt'.format(datapath))
            omics3 = np.transpose(omics3)
            omics3 = normalize(omics3, axis=0, norm='max')

            omics = np.concatenate((omics1, omics2, omics3), axis=1)

            data = omics
            #input_dim = data.shape[1]

            encoding1_dim1 = 100
            encoding2_dim1 = 50
            if typenum==15:
                middle_dim1 = 5
            elif typenum==10:
                middle_dim1 = 4
            elif typenum==5:
                middle_dim1 = 3
            dims1 = [encoding1_dim1, encoding2_dim1, middle_dim1]
            ae1 = VAE(omics1, dims1)
            ae1.train()
            ae1.autoencoder.summary()
            encoded_factor1 = ae1.predict(omics1)

            encoding1_dim2 = 80
            encoding2_dim2 = 50
            if typenum==15:
                middle_dim2 = 5
            elif typenum==10:
                middle_dim2 = 3
            elif typenum==5:
                middle_dim2 = 1
            
            dims2 = [encoding1_dim2, encoding2_dim2, middle_dim2]
            ae2 = VAE(omics2, dims2)
            ae2.train()
            ae2.autoencoder.summary()
            encoded_factor2 = ae2.predict(omics2)

            encoding1_dim3 = 80
            encoding2_dim3 = 50
            if typenum==15:
                middle_dim3 = 5
            elif typenum==10:
                middle_dim3 = 3
            elif typenum==5:
                middle_dim3 = 1
            
            dims3 = [encoding1_dim3, encoding2_dim3, middle_dim3]
            ae3 = VAE(omics3, dims3)
            ae3.autoencoder.summary()
            ae3.train()
            encoded_factor3 = ae3.predict(omics3)

            encoded_factors = np.concatenate((encoded_factor1, encoded_factor2, encoded_factor3), axis=1)

            # if not os.path.exists("{}/AE_FAETC_EM.txt".format(resultpath)):
            #     os.mknod("{}/AE_FAETC_EM.txt".format(resultpath))
            np.savetxt("{resultpath}/SVAE_FAETC_EM_{typenum}.txt".format(resultpath=resultpath,typenum=typenum), encoded_factors)








