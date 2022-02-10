from keras.layers import Input, Dense,Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize

class VAE:
    def __init__(self,data,dims):
        self.data = data

        self.original_dim = data.shape[1]
        self.intermediate_dim1 = dims[0]
        self.intermediate_dim2 = dims[1]
        self.latent_dim = dims[2]
        self.epsilon_std = 1.0

        self.x = Input(shape=(self.original_dim,))
        self.h = Dense(self.intermediate_dim1, activation='relu')(self.x)
        self.h = Dense(self.intermediate_dim2, activation='relu')(self.h)
        # 算p(Z|X)的均值和方差
        self.z_mean = Dense(self.latent_dim)(self.h)
        self.z_log_var = Dense(self.latent_dim)(self.h)

        # 参数复现技巧
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                                      stddev=self.epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        # 重参数层，相当于给输入加入噪声
        self.z = Lambda(sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # 解码层，也就是生成器部分
        self.decoder_h = Dense(self.intermediate_dim2, activation='relu')
        self.h_decoded = self.decoder_h(self.z)
        self.x_decoded_mean = Dense(self.intermediate_dim1, activation='relu')(self.h_decoded)
        self.x_decoded_mean = Dense(self.original_dim, activation='tanh')(self.x_decoded_mean)
        # self.decoder_mean = Dense(self.original_dim, activation='tanh')
        #
        # self.x_decoded_mean = self.decoder_mean(self.h_decoded)

        # 端到端的vae模型
        self.autoencoder = Model(self.x, self.x_decoded_mean)
        # 构建encoder，然后观察各个数字在隐空间的分布
        self.encoder = Model(self.x, self.z_mean)

        def vae_loss(inputs, decoded):
            xent_loss = K.sum(K.binary_crossentropy(inputs, decoded), axis=1)
            kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        # # xent_loss是重构loss，kl_loss是KL loss
        # xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        # kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # vae_loss = K.mean(xent_loss + kl_loss)
        # add_loss是新增的方法，用于更灵活地添加各种lossvae.add_loss(vae_loss)
        self.autoencoder.compile(optimizer='adam', loss=vae_loss)

    def train(self):
        self.autoencoder.fit(self.data, self.data, epochs=100,verbose=1, batch_size=16, shuffle=True)

    def predict(self,data):
        return self.encoder.predict(data)



