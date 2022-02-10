from keras.layers import Input, Dense
from keras.models import Model

class AE:
    def __init__(self,data,dims):
        self.data = data
        self.input_dim = data.shape[1]
        self.encoding1_dim = dims[0]
        self.encoding2_dim = dims[1]
        self.middle_dim = dims[2]
        # this is our input placeholder
        self.input_factor = Input(shape=(self.input_dim,))
        # 编码层
        self.encoded = Dense(self.encoding1_dim, activation='relu')(self.input_factor)
        self.encoded = Dense(self.encoding2_dim, activation='relu')(self.encoded)
        self.encoder_output = Dense(self.middle_dim)(self.encoded)

        # 解码层
        self.decoded = Dense(self.encoding2_dim, activation='relu')(self.encoder_output)
        self.decoded = Dense(self.encoding1_dim, activation='relu')(self.decoded)
        self.decoded = Dense(self.input_dim, activation='tanh')(self.decoded)

        # 构建自编码模型
        self.autoencoder = Model(inputs=self.input_factor, outputs=self.decoded)

        # 构建编码模型
        self.encoder = Model(inputs=self.input_factor, outputs=self.encoder_output)

        # compile autoencoder
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def train(self):
        self.autoencoder.fit(self.data, self.data, epochs=100,verbose=1, batch_size=8, shuffle=True)

    def predict(self,data):
        return self.encoder.predict(data)



