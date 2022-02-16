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
from ZVAEclass2 import ZVAE
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
    # omics = np.concatenate((omics1, omics2), axis=1)
    dim1=omics1.shape[1]
    dim2=omics2.shape[1]
    omics=[omics1,omics2]
    dims = [dim1, dim2]

    data = omics
    # input_dim = data.shape[1]
    # encoding1_dim = 4096
    # encoding2_dim = 1024
    # middle_dim = 2
    # dims = [encoding1_dim, encoding2_dim, middle_dim]
    vae = ZVAE(dims)
    vae.autoencoder.summary()
    vae.autoencoder.fit(omics, omics, epochs=100,verbose=1, batch_size=16, shuffle=True)
    encoded_factors = vae.encoder.predict(omics)
    if not os.path.exists("{}MMDVAE_EM.txt".format(resultpath)):
        os.mknod("{}/MMDVAE_EM.txt".format(resultpath))
    np.savetxt("{}/MMDVAE_EM.txt".format(resultpath), encoded_factors)









