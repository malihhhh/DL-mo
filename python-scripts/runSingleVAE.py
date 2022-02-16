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
from VAEclass import VAE
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
    omics = np.concatenate((omics1, omics2), axis=1)

    data = omics
    # input_dim = data.shape[1]
    encoding1_dim = 4096
    encoding2_dim = 1024
    middle_dim = 2
    dims = [encoding1_dim, encoding2_dim, middle_dim]
    vae = VAE(data, dims)
    vae.autoencoder.summary()
    vae.train()
    encoded_factors = vae.predict(data)
    if not os.path.exists("{}/VAE_FCTAE_EM.txt".format(resultpath)):
        os.mknod("{}/VAE_FCTAE_EM.txt".format(resultpath))
    np.savetxt("{}/VAE_FCTAE_EM.txt".format(resultpath), encoded_factors)









