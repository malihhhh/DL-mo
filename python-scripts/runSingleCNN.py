import numpy as np
from sklearn.preprocessing import normalize
from keras.layers import Input, Dense,concatenate,Dropout,average
from keras.models import Model
from keras import backend as K
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.layers import *
from keras.models import Model
import keras
from sklearn.metrics import classification_report
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# 训练三个神经网络
def build_NN_model1(omics, class_num):
    omics1 = omics[0]
    omics2 = omics[1]
    input1_dim = omics1.shape[1]
    input2_dim = omics2.shape[1]
    # class_num = 4

    # omics1
    input_factor1 = Input(shape=(input1_dim,), name='omics1')
    input_re1 = Reshape((-1, 1))(input_factor1)
    omics1_cnn = Conv1D(32, (300), activation='relu')(input_re1)
    omics1_cnn = MaxPool1D(100)(omics1_cnn)

    flatten1 = Flatten()(omics1_cnn)

    # omics2
    input_factor2 = Input(shape=(input2_dim,), name='omics2')
    input_re2 = Reshape((-1, 1))(input_factor2)
    omics2_cnn = Conv1D(32, (100), activation='relu', name='omics2_cnn_1')(input_re2)
    omics2_cnn = MaxPool1D(50)(omics2_cnn)

    flatten2 = Flatten(name='flatten2')(omics2_cnn)




    mid_concat = concatenate([flatten1, flatten2])
    # classifier
    nn_classifier = Dense(100, activation='relu')(mid_concat)
    nn_classifier = Dropout(0.1)(nn_classifier)
    nn_classifier = Dense(50, activation='relu')(nn_classifier)
    nn_classifier = Dropout(0.1)(nn_classifier)
    # nn_classifier = Dense(50, activation='relu')(nn_classifier)
    # nn_classifier = Dropout(0.1)(nn_classifier)
    nn_classifier = Dense(10, activation='relu')(nn_classifier)
    # nn_classifier = Dropout(0.1)(nn_classifier)
    nn_classifier = Dense(class_num, activation='softmax', name='classifier')(nn_classifier)
    my_metrics = {
        'classifier': ['acc']
    }
    my_loss = {
        'classifier': 'categorical_crossentropy', \
        }
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    zlyNN = Model(inputs=[input_factor1, input_factor2], outputs=nn_classifier)
    zlyNN.compile(optimizer=adam, loss=my_loss, metrics=my_metrics)
    return zlyNN


def build_NN_model2(omics, class_num):
    input_dim = omics.shape[1]

    # class_num = 5

    # omics1
    input_factor1 = Input(shape=(input_dim,), name='omics')
    input_re = Reshape((-1, 1))(input_factor1)
    omics1_cnn = Conv1D(32, (1000), activation='relu')(input_re)
    omics1_cnn = MaxPool1D(100)(omics1_cnn)
    omics1_cnn = Conv1D(16, (50), activation='relu')(omics1_cnn)
    omics1_cnn = MaxPool1D(10)(omics1_cnn)
    flatten = Flatten()(omics1_cnn)
    # NN
    # omics1_nn = Dense(500, activation='relu')(input_factor1)
    # omics1_nn = Dropout(0.1)(omics1_nn)
    # omics1_nn = Dense(100, activation='relu')(omics1_nn)
    # omics1_nn = Dropout(0.1)(omics1_nn)

    nn_classifier = Dense(50, activation='relu')(flatten)
    # nn_classifier = Dropout(0.1)(nn_classifier)
    if class_num == 2:
        nn_classifier = Dense(1, activation='sigmoid', name='classifier')(nn_classifier)
    else:
        nn_classifier = Dense(class_num, activation='softmax', name='classifier')(nn_classifier)
    my_metrics_multi = {
        'classifier': ['acc']
    }
    my_loss_multi = {
        'classifier': 'categorical_crossentropy', \
        }
    my_metrics_bi = {
        'classifier': ['acc']
    }
    my_loss_bi = {
        'classifier': 'binary_crossentropy', \
        }
    # compile autoencoder
    # self.autoencoder.compile(optimizer='adam', loss='mse')
    zlyNN = Model(inputs=[input_factor1], outputs=nn_classifier)
    if class_num == 2:
        zlyNN.compile(optimizer='adam', loss=my_loss_bi, metrics=my_metrics_bi)
    else:
        zlyNN.compile(optimizer='adam', loss=my_loss_multi, metrics=my_metrics_multi)
    return zlyNN



if __name__ == '__main__':
    

    # datatypes=["equal","heterogeneous"]
    # typenums=[5,10,15]
    # noise_factor=0.5
    # savepath='./result/simulations/lfnn_res.txt'
    # with open(savepath, 'w') as f2:
    #     for datatype in datatypes:
    #         f2.write(datatype+'\n')
    #         for typenum in typenums:
    #             f2.write(str(typenum)+'\n')
    #             datapath='data/simulations/{}/{}'.format(datatype, typenum)
    #             resultpath='result/simulations/{}/{}'.format(datatype, typenum)
    #             labels = np.loadtxt('{}/c.txt'.format(datapath))
    #             # groundtruth = list(np.int_(groundtruth))

    #             omics1 = np.loadtxt('{}/o1.txt'.format(datapath))
    #             omics1 = np.transpose(omics1)
    #             omics1 = normalize(omics1, axis=0, norm='max')

    #             omics2 = np.loadtxt('{}/o2.txt'.format(datapath))
    #             omics2 = np.transpose(omics2)
    #             omics2 = normalize(omics2, axis=0, norm='max')

    #             omics3 = np.loadtxt('{}/o3.txt'.format(datapath))
    #             omics3 = np.transpose(omics3)
    #             omics3 = normalize(omics3, axis=0, norm='max')

    #             omics = np.concatenate((omics1, omics2, omics3), axis=1)

    #             # k折交叉验证
    #             all_acc = []
    #             all_f1_macro = []
    #             all_f1_weighted = []

                
    #             kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    #             for train_ix, test_ix in kfold.split(omics, labels):
                    
    #                 omics_tobuild=[omics1,omics2,omics3]
    #                 train_X_1=omics1[train_ix]
    #                 train_X_2=omics2[train_ix]
    #                 train_X_3=omics3[train_ix]

    #                 test_X_1=omics1[test_ix]
    #                 test_X_2=omics2[test_ix]
    #                 test_X_3=omics3[test_ix]
    #                 # select rows
    #                 train_X, test_X = [train_X_1,train_X_2,train_X_3],[test_X_1,test_X_2,test_X_3]
    #                 #train_X, test_X = (train_X_1,train_X_2,train_X_3),(test_X_1,test_X_2,test_X_3)
    #                 train_y, test_y = labels[train_ix], labels[test_ix]
    #                 # summarize train and test composition
    #                 unique, count = np.unique(train_y, return_counts=True)
    #                 train_data_count = dict(zip(unique, count))
    #                 print('train:' + str(train_data_count))
    #                 unique, count = np.unique(test_y, return_counts=True)
    #                 test_data_count = dict(zip(unique, count))
    #                 print('test:' + str(test_data_count))

    #                 class_num=typenum
    #                 # 多分类的输出
    #                 train_y = list(np.int_(train_y))
    #                 # groundtruth = np.int_(groundtruth)
    #                 y = []
    #                 num = len(train_y)
    #                 for i in range(num):
    #                     tmp = np.zeros(class_num, dtype='uint8')
    #                     tmp[train_y[i]] = 1
    #                     y.append(tmp)
    #                 train_y = np.array(y)

    #                 test_y = list(np.int_(test_y))
    #                 # groundtruth = np.int_(groundtruth)
    #                 y = []
    #                 num = len(test_y)
    #                 for i in range(num):
    #                     tmp = np.zeros(class_num, dtype='uint8')
    #                     tmp[test_y[i]] = 1
    #                     y.append(tmp)
    #                 test_y = np.array(y)

    #                 model = build_NN_model1(omics_tobuild,class_num)
    #                 history = model.fit(train_X, train_y, epochs=50, verbose=2, batch_size=16, shuffle=True,validation_data=(test_X, test_y))
    #                 y_true = []
    #                 for i in range(len(test_y)):
    #                     y_true.append(np.argmax(test_y[i]))
    #                 predictions = model.predict(test_X)
    #                 y_pred = []
    #                 for i in range(len(predictions)):
    #                     y_pred.append(np.argmax(predictions[i]))
    #                 acc = accuracy_score(y_true, y_pred)
    #                 f1_macro = f1_score(y_true, y_pred, average='macro')
    #                 # f1_micro=f1_score(y_true, y_pred, average='micro')
    #                 f1_weighted = f1_score(y_true, y_pred, average='weighted')
    #                 all_acc.append(acc)
    #                 all_f1_macro.append(f1_macro)
    #                 all_f1_weighted.append(f1_weighted)


    #                 print(classification_report(y_true, y_pred))
    #                 # print_precison_recall_f1(y_true, y_pred)
    #             print('caicai' * 20)
    #             print(
    #                 'acc:{all_acc}\nf1_macro:{all_f1_macro}\nf1_weighted:{all_f1_weighted}\n'. \
    #                 format(all_acc=all_acc, all_f1_macro=all_f1_macro, all_f1_weighted=all_f1_weighted))
    #             avg_acc = np.mean(all_acc)
    #             avg_f1_macro = np.mean(all_f1_macro)
    #             avg_f1_weighted = np.mean(all_f1_weighted)

    #             print(
    #                 'acc:{avg_acc}\nf1_macro:{avg_f1_macro}\nf1_weighted:{avg_f1_weighted}\n'. \
    #                 format(avg_acc=avg_acc, avg_f1_macro=avg_f1_macro, avg_f1_weighted=avg_f1_weighted))
    #             f2.write('acc:{avg_acc}\nf1_macro:{avg_f1_macro}\nf1_weighted:{avg_f1_weighted}\n'. \
    #                 format(avg_acc=avg_acc, avg_f1_macro=avg_f1_macro, avg_f1_weighted=avg_f1_weighted))
    #         f2.write('*'*20)


    

    
    # groundtruth = np.loadtxt('{}/c.txt'.format(datapath))
    # groundtruth = list(np.int_(groundtruth))

    
    savepath='./result/single-cell/efcnn_res.txt'
    with open(savepath, 'w') as f2:
        datapath = 'data/single-cell/'
        resultpath = 'result/single-cell/'
        labels = np.loadtxt('{}/c.txt'.format(datapath))
        # groundtruth = list(np.int_(groundtruth))

        omics = np.loadtxt('{}/omics.txt'.format(datapath))
        omics = np.transpose(omics)
        omics1=omics[0:206]
        omics2=omics[206:412]
        omics1 = normalize(omics1, axis=0, norm='max')
        omics2 = normalize(omics2, axis=0, norm='max')
        omics = np.concatenate((omics1, omics2), axis=1)


        # k折交叉验证
        all_acc = []
        all_f1_macro = []
        all_f1_weighted = []

        
        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
        for train_ix, test_ix in kfold.split(omics, labels):
            

            train_X, test_X = omics[train_ix], omics[test_ix]
            train_y, test_y = labels[train_ix], labels[test_ix]
            # summarize train and test composition
            unique, count = np.unique(train_y, return_counts=True)
            train_data_count = dict(zip(unique, count))
            print('train:' + str(train_data_count))
            unique, count = np.unique(test_y, return_counts=True)
            test_data_count = dict(zip(unique, count))
            print('test:' + str(test_data_count))

            class_num=3
            # 多分类的输出
            train_y = list(np.int_(train_y))
            # groundtruth = np.int_(groundtruth)
            y = []
            num = len(train_y)
            for i in range(num):
                tmp = np.zeros(class_num, dtype='uint8')
                tmp[train_y[i]] = 1
                y.append(tmp)
            train_y = np.array(y)

            test_y = list(np.int_(test_y))
            # groundtruth = np.int_(groundtruth)
            y = []
            num = len(test_y)
            for i in range(num):
                tmp = np.zeros(class_num, dtype='uint8')
                tmp[test_y[i]] = 1
                y.append(tmp)
            test_y = np.array(y)

            model = build_NN_model2(omics, class_num)
            history = model.fit(train_X, train_y, epochs=50, verbose=2, batch_size=8, shuffle=True,
                                validation_data=(test_X, test_y))
            y_true = []
            for i in range(len(test_y)):
                y_true.append(np.argmax(test_y[i]))
            predictions = model.predict(test_X)
            y_pred = []
            for i in range(len(predictions)):
                y_pred.append(np.argmax(predictions[i]))
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            # f1_micro=f1_score(y_true, y_pred, average='micro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            all_acc.append(acc)
            all_f1_macro.append(f1_macro)
            all_f1_weighted.append(f1_weighted)


            print(classification_report(y_true, y_pred))
            # print_precison_recall_f1(y_true, y_pred)
        print('caicai' * 20)
        print(
            'acc:{all_acc}\nf1_macro:{all_f1_macro}\nf1_weighted:{all_f1_weighted}\n'. \
            format(all_acc=all_acc, all_f1_macro=all_f1_macro, all_f1_weighted=all_f1_weighted))
        avg_acc = np.mean(all_acc)
        avg_f1_macro = np.mean(all_f1_macro)
        avg_f1_weighted = np.mean(all_f1_weighted)

        print(
            'acc:{avg_acc}\nf1_macro:{avg_f1_macro}\nf1_weighted:{avg_f1_weighted}\n'. \
            format(avg_acc=avg_acc, avg_f1_macro=avg_f1_macro, avg_f1_weighted=avg_f1_weighted))
        f2.write('acc:{avg_acc}\nf1_macro:{avg_f1_macro}\nf1_weighted:{avg_f1_weighted}\n'. \
            format(avg_acc=avg_acc, avg_f1_macro=avg_f1_macro, avg_f1_weighted=avg_f1_weighted))


    # savepath='./result/single-cell/lfcnn_res1.txt'
    # with open(savepath, 'w') as f2:
    #     datapath = 'data/single-cell/'
    #     resultpath = 'result/single-cell/'
    #     labels = np.loadtxt('{}/c.txt'.format(datapath))
    #     # groundtruth = list(np.int_(groundtruth))
    #
    #     omics = np.loadtxt('{}/omics.txt'.format(datapath))
    #     omics = np.transpose(omics)
    #     omics1=omics[0:206]
    #     omics2=omics[206:412]
    #     omics1 = normalize(omics1, axis=0, norm='max')
    #     omics2 = normalize(omics2, axis=0, norm='max')
    #     omics = np.concatenate((omics1, omics2), axis=1)
    #
    #
    #     # k折交叉验证
    #     all_acc = []
    #     all_f1_macro = []
    #     all_f1_weighted = []
    #
    #
    #     kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    #     for train_ix, test_ix in kfold.split(omics, labels):
    #
    #         omics_tobuild=[omics1,omics2]
    #         train_X_1=omics1[train_ix]
    #         train_X_2=omics2[train_ix]
    #
    #         test_X_1=omics1[test_ix]
    #         test_X_2=omics2[test_ix]
    #
    #         # select rows
    #         train_X, test_X = [train_X_1,train_X_2],[test_X_1,test_X_2]
    #         train_y, test_y = labels[train_ix], labels[test_ix]
    #         # summarize train and test composition
    #         unique, count = np.unique(train_y, return_counts=True)
    #         train_data_count = dict(zip(unique, count))
    #         print('train:' + str(train_data_count))
    #         unique, count = np.unique(test_y, return_counts=True)
    #         test_data_count = dict(zip(unique, count))
    #         print('test:' + str(test_data_count))
    #
    #         class_num=3
    #         # 多分类的输出
    #         train_y = list(np.int_(train_y))
    #         # groundtruth = np.int_(groundtruth)
    #         y = []
    #         num = len(train_y)
    #         for i in range(num):
    #             tmp = np.zeros(class_num, dtype='uint8')
    #             tmp[train_y[i]] = 1
    #             y.append(tmp)
    #         train_y = np.array(y)
    #
    #         test_y = list(np.int_(test_y))
    #         # groundtruth = np.int_(groundtruth)
    #         y = []
    #         num = len(test_y)
    #         for i in range(num):
    #             tmp = np.zeros(class_num, dtype='uint8')
    #             tmp[test_y[i]] = 1
    #             y.append(tmp)
    #         test_y = np.array(y)
    #
    #         model = build_NN_model1(omics_tobuild,class_num)
    #         history = model.fit(train_X, train_y, epochs=50, verbose=2, batch_size=32, shuffle=True,validation_data=(test_X, test_y))
    #         y_true = []
    #         for i in range(len(test_y)):
    #             y_true.append(np.argmax(test_y[i]))
    #         predictions = model.predict(test_X)
    #         y_pred = []
    #         for i in range(len(predictions)):
    #             y_pred.append(np.argmax(predictions[i]))
    #         acc = accuracy_score(y_true, y_pred)
    #         f1_macro = f1_score(y_true, y_pred, average='macro')
    #         # f1_micro=f1_score(y_true, y_pred, average='micro')
    #         f1_weighted = f1_score(y_true, y_pred, average='weighted')
    #         all_acc.append(acc)
    #         all_f1_macro.append(f1_macro)
    #         all_f1_weighted.append(f1_weighted)
    #
    #
    #         print(classification_report(y_true, y_pred))
    #         break
    #         # print_precison_recall_f1(y_true, y_pred)
    #     print('caicai' * 20)
    #     print(
    #         'acc:{all_acc}\nf1_macro:{all_f1_macro}\nf1_weighted:{all_f1_weighted}\n'. \
    #         format(all_acc=all_acc, all_f1_macro=all_f1_macro, all_f1_weighted=all_f1_weighted))
    #     avg_acc = np.mean(all_acc)
    #     avg_f1_macro = np.mean(all_f1_macro)
    #     avg_f1_weighted = np.mean(all_f1_weighted)
    #
    #     print(
    #         'acc:{avg_acc}\nf1_macro:{avg_f1_macro}\nf1_weighted:{avg_f1_weighted}\n'. \
    #         format(avg_acc=avg_acc, avg_f1_macro=avg_f1_macro, avg_f1_weighted=avg_f1_weighted))
    #     f2.write('acc:{avg_acc}\nf1_macro:{avg_f1_macro}\nf1_weighted:{avg_f1_weighted}\n'. \
    #         format(avg_acc=avg_acc, avg_f1_macro=avg_f1_macro, avg_f1_weighted=avg_f1_weighted))
    #


    

