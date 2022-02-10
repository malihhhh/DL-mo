from keras.layers import Input, Dense
from keras.models import Model
from AEclass import AE
from keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize

class DAE(AE):
    def __init__(self,data,dims,noise_factor):
        AE.__init__(self,data,dims)
        self.noise_factor=noise_factor
        self.data_noisy=self.data+noise_factor * np.random.normal(0.0, 1.0, self.data.shape)
    def train(self):
        self.autoencoder.fit(self.data_noisy, self.data, epochs=100,verbose=1, batch_size=16, shuffle=True)

    def predict(self,data):
        return self.encoder.predict(data)



