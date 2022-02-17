""" Example for MOGONET classification
"""
from train_test import train_test

# #simulations
if __name__ == "__main__":
    datatypes=["equal","heterogeneous"]
    datatypes=["heterogeneous"]
    typenums=[5,10,15]
    typenums=[15]
    for datatype in datatypes:
        for typenum in typenums:
            datapath='simulations/{}/{}'.format(datatype, typenum)    
            data_folder = datapath
            view_list = [1,2,3]
            num_epoch_pretrain = 1000
            num_epoch = 2500
            lr_e_pretrain = 1e-3
            lr_e = 5e-4
            lr_c = 1e-3
            
            # if data_folder == 'ROSMAP':
            #     num_class = 2
            # if data_folder == 'BRCA':
            #     num_class = 5
            # if data_folder == 'gbm'or data_folder =='breast2':
            #     num_class = 4
            num_class=typenum
            train_test(data_folder, view_list, num_class,
                    lr_e_pretrain, lr_e, lr_c, 
                    num_epoch_pretrain, num_epoch) 
            print(datatype+str(typenum)+'\n')
            print('*'*100)

# single
# if __name__ == "__main__":
#
#
#     data_folder = 'single-cell'
#     view_list = [1,2]
#     num_epoch_pretrain = 500
#     num_epoch = 500
#     lr_e_pretrain = 1e-3
#     lr_e = 5e-4
#     lr_c = 1e-3
#
#     # if data_folder == 'ROSMAP':
#     #     num_class = 2
#     # if data_folder == 'BRCA':
#     #     num_class = 5
#     # if data_folder == 'gbm'or data_folder =='breast2':
#     #     num_class = 4
#     num_class=3
#     train_test(data_folder, view_list, num_class,
#             lr_e_pretrain, lr_e, lr_c,
#             num_epoch_pretrain, num_epoch)
               