from keras.layers import Input, Dense,Lambda,Concatenate
from keras import regularizers
from keras.models import Model
from keras import objectives,backend as K
from keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from keras.optimizers import Adam, RMSprop
import tensorflow as tf

class ZVAE:
    def __init__(self,original_dims):
        #单细胞数据
        #组学一编码器维度
        self.o1_original_dim=original_dims[0]
        self.o1_nn1_dim=1000
        self.o1_nn2_dim=100
        #组学二编码器维度
        self.o2_original_dim=original_dims[1]
        self.o2_nn1_dim=1000
        self.o2_nn2_dim=100
        

        self.share_dim=100
        self.latent_dim=10

        

        self.share_dim=100
        self.latent_dim=15

        acti='relu'
        self.reg_lambda=0
        self.epsilon_std = 1.0

        #组学一编码器
        self.omics1 = Input(shape=(self.o1_original_dim,),name='input_omics1')
        self.o1_encoder = Dense(self.o1_nn1_dim, name='o1_encoder_layer1', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.omics1)
        self.o1_encoder = Dense(self.o1_nn2_dim, name='o1_encoder_layer2', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.o1_encoder)
        #组学二编码器
        self.omics2 = Input(shape=(self.o2_original_dim,),name='input_omics2')
        self.o2_encoder = Dense(self.o2_nn1_dim, name='o2_encoder_layer1', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.omics2)
        self.o2_encoder = Dense(self.o2_nn2_dim, name='o2_encoder_layer2', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.o2_encoder)
        
        #三个组学中间输出拼接
        self.concat = Concatenate(axis=-1, name='concat_layer')([self.o1_encoder, self.o2_encoder])
        self.share = Dense(self.share_dim, name='shared_layer', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.concat)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, self.latent_dim), mean=0., stddev=1)
            return z_mean_ + K.exp(0.5 * z_log_var_) * epsilon
        #均值
        self.z_mean = Dense(self.latent_dim, name='z_mean', activation='linear')(self.share)
        #方差
        self.z_log_var = Dense(self.latent_dim, name='z_log_var', activation='linear')(self.share)
        #采样
        self.z=Lambda(sampling, output_shape=(self.latent_dim,), name='lambda')([self.z_mean, self.z_log_var])

        #组学一解码器
        self.o1_decoder = Dense(self.o1_nn2_dim, name='o1_decoder_layer2', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.z)
        self.o1_decoder = Dense(self.o1_nn1_dim, name='o1_decoder_layer1', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.o1_decoder)
        self.o1_decoded = Dense(self.o1_original_dim, name='o1_decoded', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.o1_decoder)

        #组学二解码器
        self.o2_decoder = Dense(self.o2_nn2_dim, name='o2_decoder_layer2', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.z)
        self.o2_decoder = Dense(self.o2_nn1_dim, name='o2_decoder_layer1', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.o2_decoder)
        self.o2_decoded = Dense(self.o2_original_dim, name='o2_decoded', activation=acti, kernel_regularizer=regularizers.l2(self.reg_lambda))(self.o2_decoder)

        

        #构造整个自编码器模型
        self.autoencoder = Model(inputs=[self.omics1, self.omics2], outputs=[self.o1_decoded,self.o2_decoded])
        #kl散度损失函数
        def vae_mse_loss(x, x_decoded_mean):
            mse_loss = objectives.mse(x, x_decoded_mean)
            #kl_loss = - 0.5 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
            kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
            return mse_loss + kl_loss
        def loss1(inputs, decoded):
            xent_loss = K.sum(K.binary_crossentropy(inputs, decoded), axis=1)
            return xent_loss
        def vae_loss(inputs, decoded):
            xent_loss = K.sum(K.binary_crossentropy(inputs, decoded), axis=1)
            kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        #self.autoencoder.add_loss(kl_loss)
        # self.autoencoder.compile(optimizer=Adam(1e-5),
        #                          loss=['mae','mae','mae'])
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
            return K.mean(mse_loss+loss_mmd)
        self.autoencoder.compile(optimizer=Adam(1e-5),loss=['mse',vae_loss_mmd])
        # print(self.autoencoder.summary())
        self.encoder = Model(inputs=[self.omics1, self.omics2],outputs=self.z_mean)


if __name__ == '__main__':
    datapath = 'data/BRCA'
    omics1 = np.loadtxt('{}/1_all.csv'.format(datapath),delimiter=',')
    #omics1 = np.transpose(omics1)
    #omics1 = normalize(omics1, axis=0, norm='max')
    print(omics1.shape)
    omics2 = np.loadtxt('{}/2_all.csv'.format(datapath),delimiter=',')
    #omics2 = np.transpose(omics2)
    #omics2 = normalize(omics2, axis=0, norm='max')
    print(omics2.shape)
    omics3 = np.loadtxt('{}/3_all.csv'.format(datapath),delimiter=',')
    #omics3 = np.transpose(omics3)
    #omics3 = normalize(omics3, axis=0, norm='max')
    print(omics3.shape)
    omics=[omics1,omics2,omics3]

    vae = ZVAE()
    vae.autoencoder.summary()
    vae.autoencoder.fit(omics, omics, epochs=100,verbose=1, batch_size=16, shuffle=True)
    encoded_factors = vae.encoder.predict(omics)
    resultpath='result/nmf/'
    np.savetxt("{}/zly_vae_em_mse_mmd_100ep.txt".format(resultpath), encoded_factors)






