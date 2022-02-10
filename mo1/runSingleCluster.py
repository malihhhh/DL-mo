import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize



# data_names = ['VAE_FCTAE_EM','AE_FAETC_EM', 'AE_FCTAE_EM', 'DAE_FAETC_EM', 'DAE_FCTAE_EM', 'LSTMVAE_FCTAE_EM']
data_names = ['SVAE_FCTAE_EM','MMDVAE_EM']
for data_name in data_names:
    # encoded_factors=np.loadtxt('./result/cancer_do_cluster/{f}/{d}.txt'.format(f=f, d=data_name))
    encoded_factors=np.loadtxt('./result/single-cell/{d}.txt'.format(d=data_name))
    savepath='./result/single-cell/{d}_cluster_result.txt'.format(d=data_name)
    with open(savepath, 'w') as f2:
        print('method:{d}\n'.format(d=data_name))
        f2.write('method:{d}\n'.format(d=data_name))
        for typenum in range(2,7,1):
            all_silhouette=[]
            all_DBI=[]
            for i in range(100):
                clf = KMeans(n_clusters=typenum)
                clf.fit(encoded_factors)  # 模型训练
                labels = clf.labels_
                silhouetteScore = silhouette_score(encoded_factors, labels, metric='euclidean')
                all_silhouette.append(silhouetteScore)
                davies_bouldinScore = davies_bouldin_score(encoded_factors, labels)
                all_DBI.append(davies_bouldinScore)
            avg_silhouette=np.mean(all_silhouette)
            avg_DBI=np.mean(all_DBI)

            # print("silhouetteScore:", avg_silhouette)
            # print("davies_bouldinScore:", avg_DBI)
            print('k:{k}\nsilhouetteScore:{s}\ndavies_bouldinScore:{d}\n'.format(k=typenum, s=avg_silhouette,d=avg_DBI))
            f2.write('*'*20+'\n')
            f2.write('k:{k}\nsilhouetteScore:{s}\ndavies_bouldinScore:{d}\n'.format(k=typenum, s=avg_silhouette,d=avg_DBI))


#直接拼接
# files = ['aml', 'breast', 'colon', 'kidney', 'liver', 'lung', 'melanoma', 'ovarian', 'sarcoma','gbm']
# for f in files:
#     datapath='./data/cancer_do_cluster/{f}'.format(f=f)
#     omics1 = np.loadtxt('{}/log_exp_omics.txt'.format(datapath))
#     omics1 = np.transpose(omics1)
#     omics1 = normalize(omics1, axis=0, norm='max')
#     print(omics1.shape)
#     omics2 = np.loadtxt('{}/log_mirna_omics.txt'.format(datapath))
#     omics2 = np.transpose(omics2)
#     omics2 = normalize(omics2, axis=0, norm='max')
#     print(omics2.shape)
#     omics3 = np.loadtxt('{}/methy_omics.txt'.format(datapath))
#     omics3 = np.transpose(omics3)
#     omics3 = normalize(omics3, axis=0, norm='max')
#     print(omics3.shape)
#     omics = np.concatenate((omics1, omics2, omics3), axis=1)
#     encoded_factors=omics
#     savepath='./result/cancer_do_cluster/{f}/Contact_cluster_result.txt'.format(f=f)
#     with open(savepath, 'w') as f2:
#         print('cancer:{f}\nmethod:直接拼接'.format(f=f))
#         f2.write('cancer:{f}\nmethod:直接拼接\n'.format(f=f))
#         for typenum in range(2,7,1):
#             all_silhouette=[]
#             all_DBI=[]
#             for i in range(100):
#                 clf = KMeans(n_clusters=typenum)
#                 clf.fit(encoded_factors)  # 模型训练
#                 labels = clf.labels_
#                 silhouetteScore = silhouette_score(encoded_factors, labels, metric='euclidean')
#                 all_silhouette.append(silhouetteScore)
#                 davies_bouldinScore = davies_bouldin_score(encoded_factors, labels)
#                 all_DBI.append(davies_bouldinScore)
#             avg_silhouette=np.mean(all_silhouette)
#             avg_DBI=np.mean(all_DBI)

#             # print("silhouetteScore:", avg_silhouette)
#             # print("davies_bouldinScore:", avg_DBI)
#             print('k:{k}\nsilhouetteScore:{s}\ndavies_bouldinScore:{d}\n'.format(k=typenum, s=avg_silhouette,d=avg_DBI))
#             f2.write('zly'*20+'\n')
#             f2.write('k:{k}\nsilhouetteScore:{s}\ndavies_bouldinScore:{d}\n'.format(k=typenum, s=avg_silhouette,d=avg_DBI))            
            