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
#from AEclass import AE
import os

if __name__ == '__main__':
    datapath = 'data/single-cell/'
    resultpath = 'result/single-cell/'
    # groundtruth = np.loadtxt('{}/c.txt'.format(datapath))
    # groundtruth = list(np.int_(groundtruth))

    omics = np.loadtxt('{}/omics.txt'.format(datapath))
    omics = np.transpose(omics)
    omics1 = omics[0:206]
    omics2 = omics[206:412]
    omics1 = normalize(omics1, axis=0, norm='max')
    omics2 = normalize(omics2, axis=0, norm='max')
    #omics = np.concatenate((omics1, omics2), axis=1)

    encoding1_dim1 = 2048
    encoding2_dim1 = 512
    middle_dim1 = 1
    dims1 = [encoding1_dim1, encoding2_dim1, middle_dim1]
    ae1 = VAE(omics1, dims1)
    ae1.train()
    ae1.autoencoder.summary()
    encoded_factor1 = ae1.predict(omics1)

    encoding1_dim2 = 2048
    encoding2_dim2 = 512
    middle_dim2 = 1
    dims2 = [encoding1_dim2, encoding2_dim2, middle_dim2]
    ae2 = VAE(omics2, dims2)
    ae2.train()
    ae2.autoencoder.summary()
    encoded_factor2 = ae2.predict(omics2)


    encoded_factors = np.concatenate((encoded_factor1, encoded_factor2), axis=1)

    if not os.path.exists("{}/SVAE_FAETC_EM.txt".format(resultpath)):
        os.mknod("{}/SVAE_FAETC_EM.txt".format(resultpath))
    np.savetxt("{}/SVAE_FAETC_EM.txt".format(resultpath), encoded_factors)









