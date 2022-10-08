# -*- coding: utf-8 -*-

import numpy as np
import os
class DefaultConfig():
    model='DAML'
    num_fea = 1
    dataset = 'Beijing'
    # -------------base config-----------------------#
    use_gpu = True
    gpu_id = 1
    multi_gpu = False
    gpu_ids = []

    seed = 2019
    num_epochs = 50
    num_workers = 0

    optimizer = 'Adam'
    output = 1e-3  # optimizer rameteri
    lr = 2e-3
    loss_method = 'mse'
    drop_out = 0.2

    use_word_embedding = True

    id_emb_size = 32
    query_mlp_size = 128
    fc_dim = 32

    doc_len = 500
    filters_num = 100
    kernel_size = 3
    num_heads = 2

    use_review = True
    use_doc = True
    self_att = True
    point_num_heads = 1

    r_id_merge = 'cat'  # review and ID feature
    ui_merge = 'cat'  # cat/add/dot
    output = 'lfm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    fine_step = False  # save mode in step level, defualt in epoch
    # pth_path = './checkpoints/' + model + '_' + dataset+'_default_rating.pth'   # the saved pth path for test
    pth_path = './checkpoints/' + model + '_' + dataset + '_' + str(num_epochs)+'.pth'
    print_opt = str(num_epochs)

    # prediction_results_path = f'prediction_results/'
    # if not os.path.exists(prediction_results_path):
    #     os.makedirs(prediction_results_path, exist_ok=True)


    batch_size = 50
    print_step = 100
    weight_decay = 0.001
    ## 读取

    def parse(self,dataset,kwargs):
        f = open('./config/' + dataset + '_para_dict_rating.txt', 'r')
        a = f.read()
        para_dict = eval(a)

        self.vocab_size = para_dict['vocab_size']
        self.r_max_len = para_dict['r_max_len']
        self.u_max_r = para_dict['u_max_r']
        self.i_max_r = para_dict['i_max_r']
        print('self.i_max_r',self.i_max_r)

        self.train_data_size = para_dict['train_data_size']
        self.test_data_size = para_dict['test_data_size']
        self.val_data_size = para_dict['val_data_size']

        self.user_num = para_dict['user_num'] + 2
        self.item_num = para_dict['item_num'] + 2
        self.travel_type_num = para_dict['travel_type_num'] + 2
        self.travel_month_num = para_dict['travel_month_num'] + 2

        self.word_dim = 300
        self.lstm_hidden_size = 100
        self.bidirectional = False
        self.num_layers = 1

        self.data_root = f'./dataset/{dataset}'
        prefix = f'{self.data_root}/train'
        print("data_root", self.data_root)

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        self.w2v_path = f'{prefix}/w2v_'+str(self.word_dim)+'.npy'
        # self.w2v_path = f'{prefix}/w2v.npy'
        '''
        user can update the default hyperparamter
        '''
        print("load npy from dist...")
        self.users_review_list = np.load(self.user_list_path, encoding='bytes')
        self.items_review_list = np.load(self.item_list_path, encoding='bytes')
        self.user2itemid_list = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_list = np.load(self.item2userid_path, encoding='bytes')
        self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))
        print('*************************************************')

