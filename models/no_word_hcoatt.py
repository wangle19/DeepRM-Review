# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.language import *
from models.attention import *

class No_WORD_HCOATT(nn.Module):
    '''
    NARRE: WWW 2018
    '''
    def __init__(self, opt):
        super(No_WORD_HCOATT, self).__init__()
        self.opt = opt
        self.num_fea = 3  # ID + Review +context
        self.head = opt.point_num_heads

        #id embedding
        self.user_id_embedding = nn.Embedding(self.opt.user_num, self.opt.id_emb_size)  # user/item num * 32
        self.item_id_embedding = nn.Embedding(self.opt.item_num, self.opt.id_emb_size)  # user/item num * 32
        self.travel_type_embedding = nn.Embedding(self.opt.travel_type_num, self.opt.id_emb_size)
        self.travel_month_embedding = nn.Embedding(self.opt.travel_month_num, self.opt.id_emb_size)
        self.user_word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300

        # item_net

        self.item_word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)  # vocab_size * 300

        # self.review_hidden_size = opt.lstm_hidden_size * (1 + int(opt.bidirectional))
        self.review_hidden_size = self.opt.word_dim
        # self.review_coatt = Co_Attention(dim=self.opt.id_emb_size, gumbel=False, pooling='avg')
        self.review_coatt = nn.ModuleList([Co_Attention(dim=self.opt.id_emb_size, gumbel=False, pooling='avg') for _ in range(self.head)])
        #
        self.relu = nn.ReLU()
        self.user_fc_layer = nn.Linear(self.review_hidden_size, self.opt.id_emb_size)
        self.item_fc_layer = nn.Linear(self.review_hidden_size, self.opt.id_emb_size)
        self.dropout = nn.Dropout(self.opt.drop_out)

        # self.fea_coatt = Co_Attention(dim=self.opt.id_emb_size, gumbel=False, pooling='avg')
        self.fea_coatt = nn.ModuleList([Co_Attention(dim=self.opt.id_emb_size, gumbel=False, pooling='avg') for _ in range(self.head)])
        # final fc
        self.u_fc = nn.Linear(self.opt.id_emb_size * self.head, self.opt.id_emb_size)
        self.i_fc = nn.Linear(self.opt.id_emb_size * self.head, self.opt.id_emb_size)

        self.reset_para()

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc, travel_type, travel_month = datas
        batch = user_reviews.size(0)

        # --------------- id embedding ----------------------------------
        user_id_emb = self.user_id_embedding(uids)
        item_id_emb = self.item_id_embedding(iids)
        travel_type_emd = self.travel_type_embedding(travel_type)
        travel_month_emd = self.travel_month_embedding(travel_month)
        feature_embedding = torch.stack((user_id_emb,item_id_emb,travel_type_emd,travel_month_emd),1)

        # --------------- word embedding ----------------------------------
        user_reviews = self.user_word_embs(user_reviews)  #(20,1,82,300)
        item_reviews = self.item_word_embs(item_reviews)  #(20,25,82,300)


        user_review_embedding = self.dropout(
            self.relu(self.user_fc_layer(torch.sum(user_reviews, 2))))  # [batch, review_hidden_size]
        item_review_embedding = self.dropout(
            self.relu(self.item_fc_layer(torch.sum(item_reviews, 2))))  # [bat



        u_fea = []
        i_fea = []
        for i in range(self.head):
            r_coatt = self.review_coatt[i]
            fea_coatt = self.fea_coatt[i]
            # --------review_co-attention --------------------
            user_review_att, item_review_att = r_coatt(user_review_embedding, item_review_embedding)
            user_att_review_embedding = user_review_embedding * (user_review_att)
            item_att_review_embedding = item_review_embedding * (item_review_att)
            review_embedding = torch.cat((user_att_review_embedding,item_att_review_embedding),1)

            # --------review_feature_co-attention --------------------
            review_att, fea_att = fea_coatt(review_embedding, feature_embedding)
            fea_att_embedding = feature_embedding * fea_att
            user_id_emb,item_id_emb,travel_type_emd,travel_month_emd = torch.split(fea_att_embedding,1, 1)
            user_id_emb = self.dropout(torch.sum(user_id_emb, 1))  # [batch, review_hidden_size]
            item_id_emb = self.dropout(torch.sum(item_id_emb, 1))  # [batch, review_hidden_size]
            travel_type_emd = self.dropout(torch.sum(travel_type_emd, 1))  # [batch, review_hidden_size]
            travel_month_emd = self.dropout(torch.sum(travel_month_emd, 1))  # [batch, review_hidden_size]

            review_att_embedding =review_embedding * review_att
            user_2att_review_embedding, item_2att_review_embedding = torch.split(review_att_embedding, [self.opt.u_max_r,self.opt.i_max_r], 1)
            user_fea = torch.sum(user_2att_review_embedding, 1)  # [batch, review_hidden_size]
            item_fea = torch.sum(item_2att_review_embedding, 1) # [batch, review_hidden_size]
            u_fea.append(user_fea)
            i_fea.append(item_fea)
        u_fea = torch.cat(u_fea, 1)
        i_fea = torch.cat(i_fea, 1)

        u_fea = self.dropout(self.u_fc(u_fea))
        i_fea = self.dropout(self.i_fc(i_fea))

        user_fea = torch.stack([user_id_emb,travel_type_emd, u_fea], 1)
        item_fea = torch.stack([item_id_emb,travel_month_emd, i_fea], 1)

        return user_fea,item_fea

    def reset_para(self):
        for fc in [self.user_fc_layer, self.item_fc_layer]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.uniform_(fc.bias, -0.1, 0.1)

        if self.opt.use_word_embedding:
            w2v = torch.from_numpy(np.load(self.opt.w2v_path))
            if self.opt.use_gpu:
                self.user_word_embs.weight.data.copy_(w2v.cuda())
                self.item_word_embs.weight.data.copy_(w2v.cuda())
            else:
                self.user_word_embs.weight.data.copy_(w2v)
                self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.xavier_normal_(self.user_word_embs.weight)
            nn.init.xavier_normal_(self.item_word_embs.weight)

        nn.init.uniform_(self.user_id_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.item_id_embedding.weight, a=-0.1, b=0.1)


