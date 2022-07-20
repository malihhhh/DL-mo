from keras.layers import Input, Dense,Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras import optimizers
from keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize


import tensorflow as tf

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

        def compute_kernel(x, y):
            x_size = tf.shape(x)[0]
            y_size = tf.shape(y)[0]
            dim = tf.shape(x)[1]
            tiled_x = tf.tile(tf.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
            tiled_y = tf.tile(tf.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
            return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

        def compute_mmd(x, y):
            x_kernel = compute_kernel(x, x)
            y_kernel = compute_kernel(y, y)
            xy_kernel = compute_kernel(x, y)
            return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

        true_samples = tf.random_normal(shape=tf.shape(self.z))

        def vae_loss_mmd(inputs, decoded):
            mse_loss = objectives.mse(inputs, decoded)
            loss_mmd = compute_mmd(true_samples, self.z)
            return K.mean(mse_loss + loss_mmd)

        self.autoencoder.compile(optimizer=optimizers.Adam(lr=0.0001), loss=vae_loss_mmd)

    def train(self):
        self.autoencoder.fit(self.data, self.data, epochs=5,verbose=1, batch_size=16, shuffle=True)

    def predict(self,data):
        return self.encoder.predict(data)



