import numpy as np
from sklearn.preprocessing import normalize
from keras.layers import Input, Dense,concatenate,Dropout,average
from keras.models import Model
from keras import backend as K
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.layers import Input, Dense,concatenate,Dropout,average
from keras.models import Model
import keras
from sklearn.metrics import classification_report

    
    

#cancer数据
if __name__ == '__main__':
    # files = ['breast2']
    files = ['gbm','breast2']
    for f in files:
        datapath='./data/cancer_d2d/{f}'.format(f=f)
        omics1 = np.loadtxt('{}/after_log_exp.txt'.format(datapath),str)
        omics1 = np.delete(omics1, 0, axis=1)
        #omics1 = np.transpose(omics1)
        omics1 = omics1.astype(np.float)
        omics1 = normalize(omics1, axis=0, norm='max')
        print(omics1.shape)

        omics2 = np.loadtxt('{}/after_log_mirna.txt'.format(datapath),str)
        omics2= np.delete(omics2, 0, axis=1)
        #omics2 = np.transpose(omics2)
        omics2 = omics2.astype(np.float)
        omics2 = normalize(omics2, axis=0, norm='max')
        print(omics2.shape)

        omics3 = np.loadtxt('{}/after_methy.txt'.format(datapath),str)
        omics3= np.delete(omics3,0,axis=1)
        #omics3 = np.transpose(omics3)
        omics3 = omics3.astype(np.float)
        omics3 = normalize(omics3, axis=0, norm='max')
        print(omics3.shape)

        labels = np.loadtxt('{datapath}/after_labels.txt'.format(datapath=datapath), str)
        labels = np.delete(labels, 0, axis=1)
        labels = labels.astype(np.int)
        labels = np.squeeze(labels,axis=1)
        # datapath = 'data/BRCA'
        # omics1 = np.loadtxt('{}/1_all.csv'.format(datapath),delimiter=',')
        # #omics1 = np.transpose(omics1)
        # omics1 = normalize(omics1, axis=0, norm='max')

        # omics2 = np.loadtxt('{}/2_all.csv'.format(datapath),delimiter=',')
        # #omics2 = np.transpose(omics2)
        # omics2 = normalize(omics2, axis=0, norm='max')

        # omics3 = np.loadtxt('{}/3_all.csv'.format(datapath),delimiter=',')
        # #omics3 = np.transpose(omics3)
        # omics3 = normalize(omics3, axis=0, norm='max')
        
        # k折交叉验证
        all_acc = []
        all_f1_macro = []
        all_f1_weighted = []
        all_auc_macro = []
        all_auc_weighted = []
        #omics = np.loadtxt('./result/nmf/mf_em.txt')
        omics = np.concatenate((omics1, omics2, omics3), axis=1)
        
        # labels = np.loadtxt('./data/BRCA/labels_all.csv', delimiter=',')
        # data=np.concatenate([])
        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
        for train_ix, test_ix in kfold.split(omics1, labels):
            omics_tobuild=[omics1,omics2,omics3]
            train_X_1=omics1[train_ix]
            train_X_2=omics2[train_ix]
            train_X_3=omics3[train_ix]

            test_X_1=omics1[test_ix]
            test_X_2=omics2[test_ix]
            test_X_3=omics3[test_ix]
            
            train_y, test_y = labels[train_ix], labels[test_ix]
            
            np.savetxt('{}/1_tr.csv'.format(datapath), train_X_1, delimiter=',')
            np.savetxt('{}/2_tr.csv'.format(datapath), train_X_2, delimiter=',')
            np.savetxt('{}/3_tr.csv'.format(datapath), train_X_3, delimiter=',')
            np.savetxt('{}/1_te.csv'.format(datapath), test_X_1, delimiter=',')
            np.savetxt('{}/2_te.csv'.format(datapath), test_X_2, delimiter=',')
            np.savetxt('{}/3_te.csv'.format(datapath), test_X_3, delimiter=',')
            np.savetxt('{}/labels_tr.csv'.format(datapath), train_y, delimiter=',')
            np.savetxt('{}/labels_te.csv'.format(datapath), test_y, delimiter=',')
            break


#simulations数据
# if __name__ == '__main__':
#     datatypes=["equal","heterogeneous"]
#     typenums=[5,10,15]
#     for datatype in datatypes:
#         for typenum in typenums:
#             datapath='data/simulations/{}/{}'.format(datatype, typenum)
    
#             labels = np.loadtxt('{}/c.txt'.format(datapath))
                

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

#             kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
#             for train_ix, test_ix in kfold.split(omics1, labels):
#                 omics_tobuild=[omics1,omics2,omics3]
#                 train_X_1=omics1[train_ix]
#                 train_X_2=omics2[train_ix]
#                 train_X_3=omics3[train_ix]

#                 test_X_1=omics1[test_ix]
#                 test_X_2=omics2[test_ix]
#                 test_X_3=omics3[test_ix]
                
#                 train_y, test_y = labels[train_ix], labels[test_ix]
                
#                 np.savetxt('{}/1_tr.csv'.format(datapath), train_X_1, delimiter=',')
#                 np.savetxt('{}/2_tr.csv'.format(datapath), train_X_2, delimiter=',')
#                 np.savetxt('{}/3_tr.csv'.format(datapath), train_X_3, delimiter=',')
#                 np.savetxt('{}/1_te.csv'.format(datapath), test_X_1, delimiter=',')
#                 np.savetxt('{}/2_te.csv'.format(datapath), test_X_2, delimiter=',')
#                 np.savetxt('{}/3_te.csv'.format(datapath), test_X_3, delimiter=',')
#                 np.savetxt('{}/labels_tr.csv'.format(datapath), train_y, delimiter=',')
#                 np.savetxt('{}/labels_te.csv'.format(datapath), test_y, delimiter=',')
#                 break

#single数据
if __name__ == '__main__':
    
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
    

    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(omics1, labels):
        omics_tobuild=[omics1,omics2]
        train_X_1=omics1[train_ix]
        train_X_2=omics2[train_ix]


        test_X_1=omics1[test_ix]
        test_X_2=omics2[test_ix]

        
        train_y, test_y = labels[train_ix], labels[test_ix]
        
        np.savetxt('{}/1_tr.csv'.format(datapath), train_X_1, delimiter=',')
        np.savetxt('{}/2_tr.csv'.format(datapath), train_X_2, delimiter=',')
        np.savetxt('{}/1_te.csv'.format(datapath), test_X_1, delimiter=',')
        np.savetxt('{}/2_te.csv'.format(datapath), test_X_2, delimiter=',')
        np.savetxt('{}/labels_tr.csv'.format(datapath), train_y, delimiter=',')
        np.savetxt('{}/labels_te.csv'.format(datapath), test_y, delimiter=',')
        break



