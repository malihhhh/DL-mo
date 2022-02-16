import numpy as np
from numpy.lib.shape_base import split

def process(datapath,omics_name):
    pass




if __name__ == '__main__':
    datapaths=['breast','gbm']
    omics_names=['log_exp','log_mirna','methy']

    # datapath='./breast'
    datapath='./gbm'

    omics1 = np.loadtxt('{path}/log_exp_omics.txt'.format(path=datapath), str)

    omics2 = np.loadtxt('{path}/log_mirna_omics.txt'.format(path=datapath), str)

    omics3 = np.loadtxt('{path}/methy_omics.txt'.format(path=datapath), str)

    


    names = omics1[0]
    # get sample names
    names = np.delete(names, 0, axis=0)
    for i in range(len(names)):
        names[i]=str(names[i]).replace('.','')
        names[i]=str(names[i]).replace('"','')
    
    # delete row names
    omics1 = np.delete(omics1, 0, axis=1)
    # delete col names
    #omics1 = np.delete(omics1, 0, axis=0)
    omics1 = np.transpose(omics1)
    for i in range(len(omics1)):
        omics1[i][0]=str(omics1[i][0]).replace('.','')
        omics1[i][0]=str(omics1[i][0]).replace('"','')


    # delete row names
    omics2 = np.delete(omics2, 0, axis=1)
    # delete col names
    #omics1 = np.delete(omics1, 0, axis=0)
    omics2 = np.transpose(omics2)
    for i in range(len(omics2)):
        omics2[i][0]=str(omics2[i][0]).replace('.','')
        omics2[i][0]=str(omics2[i][0]).replace('"','')
    
    # delete row names
    omics3 = np.delete(omics3, 0, axis=1)
    # delete col names
    #omics1 = np.delete(omics1, 0, axis=0)
    omics3 = np.transpose(omics3)
    for i in range(len(omics3)):
        omics3[i][0]=str(omics3[i][0]).replace('.','')
        omics3[i][0]=str(omics3[i][0]).replace('"','')

    print(omics1.shape)
    print(omics2.shape)
    print(omics3.shape)

    #print(omics1[1])
    #print(len(omics1))
    #subtype
    # subtype=['Luminal A','Luminal B','Basal-like','Normal-like','HER2-enriched']
    subtype=['Proneural','Classical','Mesenchymal','Neural','HER2-enriched']
    #解决\ufeff问题
    labels = np.loadtxt('{path}/{path}.csv'.format(path=datapath),str,delimiter=',',encoding='UTF-8-sig')

    for i in range(len(labels)):
        labels[i][0]=(labels[i][0]).replace('-','')
        labels[i][0]=(labels[i][0]).replace('"','')
    print('*'*20)
    labels = np.delete(labels, np.isin(labels[:,0], names, invert=True), axis=0)
    #print(np.where(np.isin(labels[:,0], names, invert=True)))
    print('*'*20)
    #labels[]
    # delete NA
    labels = np.delete(labels, np.where(labels=='NA'), axis=0)
    for i in range(len(subtype)):
        labels[labels==subtype[i]]=i
    # print(labels[0][0])
    # print(names)
    # print(names[0])
    # print(labels[0][0]==names[0])

    omics1 = np.delete(omics1, np.isin(omics1[:,0], labels[:,0], invert=True), axis=0)
    omics2 = np.delete(omics2, np.isin(omics2[:,0], labels[:,0], invert=True), axis=0)
    omics3 = np.delete(omics3, np.isin(omics3[:,0], labels[:,0], invert=True), axis=0)
    # print(labels.shape)
    # print(omics1.shape)

    #测试样本是否对齐
    # for i in range(len(omics1)):
    #     if omics1[i][0] != labels[i][0]:
    #         print('zly'*100)
    #     else:
    #         print('caicai'*100)
    
    # for i in range(len(omics2)):
    #     if omics2[i][0] != labels[i][0]:
    #         print('zly'*100)
    #     else:
    #         print('caicai'*100)
    
    # for i in range(len(omics3)):
    #     if omics3[i][0] != labels[i][0]:
    #         print('zly'*100)
    #     else:
    #         print('caicai'*100)

    np.savetxt('{path}/after_log_exp.txt'.format(path=datapath),omics1,fmt="%s")
    np.savetxt('{path}/after_log_mirna.txt'.format(path=datapath),omics2,fmt="%s")
    np.savetxt('{path}/after_methy.txt'.format(path=datapath),omics3,fmt="%s")
    np.savetxt('{path}/after_labels.txt'.format(path=datapath),labels,fmt="%s")

    





