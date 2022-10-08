# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class HCOATT_TWC(nn.Module):
    '''
    Multi-Pointer Co-Attention Network for Recommendation
    WWW 2018
    '''
    def __init__(self, opt, head=3):
        '''
        head: the number of pointers
        '''
        super(HCOATT_TWC, self).__init__()

        self.opt = opt
        self.num_fea = 3  # ID + DOC
        self.head = opt.point_num_heads
        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300

        # id embedding
        self.user_id_embedding = nn.Embedding(self.opt.user_num, self.opt.id_emb_size)  # user/item num * 32
        self.item_id_embedding = nn.Embedding(self.opt.item_num, self.opt.id_emb_size)  # user/item num * 32
        self.travel_type_embedding = nn.Embedding(self.opt.travel_type_num, self.opt.id_emb_size)
        self.travel_month_embedding = nn.Embedding(self.opt.travel_month_num, self.opt.id_emb_size)

        # review gate
        self.fc_g1 = nn.Linear(opt.word_dim, opt.word_dim)
        self.fc_g2 = nn.Linear(opt.word_dim, opt.word_dim)

        # multi points
        self.review_coatt = nn.ModuleList([Co_Attention(opt.word_dim, gumbel=True, pooling='max') for _ in range(head)])
        self.word_coatt = nn.ModuleList([Co_Attention(opt.word_dim, gumbel=False, pooling='avg') for _ in range(head)])

        self.user_fc_layer = nn.Linear(opt.word_dim, opt.id_emb_size)
        self.item_fc_layer = nn.Linear(opt.word_dim, opt.id_emb_size)

        self.fea_coatt = nn.ModuleList([Co_Attention(dim= opt.id_emb_size, gumbel=False, pooling='avg') for _ in range(self.head)])

        # final fc
        self.u_fc = nn.Linear(opt.id_emb_size * opt.point_num_heads, opt.id_emb_size)
        self.i_fc = nn.Linear(opt.id_emb_size * opt.point_num_heads, opt.id_emb_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(opt.drop_out)

        self.reset_para()

    def forward(self, datas):
        '''
        user_reviews, item_reviews, uids, iids, \
        user_item2id, item_user2id, user_doc, item_doc = datas
        :user_reviews: B * L1 * N
        :item_reviews: B * L2 * N
        '''
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc, travel_type, travel_month = datas
        # --------------- id embedding ----------------------------------
        user_id_emb = self.user_id_embedding(uids)
        item_id_emb = self.item_id_embedding(iids)
        travel_type_emd = self.travel_type_embedding(travel_type)
        travel_month_emd = self.travel_month_embedding(travel_month)
        feature_embedding = torch.stack((user_id_emb, item_id_emb, travel_type_emd, travel_month_emd), 1)

        # ------------------review-level co-attention ---------------------------------
        u_word_embs = self.user_word_embs(user_reviews)
        i_word_embs = self.item_word_embs(item_reviews)
        u_reviews = self.review_gate(u_word_embs)
        i_reviews = self.review_gate(i_word_embs)
        u_fea = []
        i_fea = []
        for i in range(self.head):
            r_coatt = self.review_coatt[i]
            w_coatt = self.word_coatt[i]
            fea_coatt = self.fea_coatt[i]

            # ------------------review-level co-attention ---------------------------------
            p_u, p_i = r_coatt(u_reviews, i_reviews)             # B * L1/2 * 1
            # ------------------word-level co-attention ---------------------------------
            u_r_words = user_reviews.permute(0, 2, 1).float().bmm(p_u)   # (B * N * L1) X (B * L1 * 1)
            i_r_words = item_reviews.permute(0, 2, 1).float().bmm(p_i)   # (B * N * L2) X (B * L2 * 1)
            u_words = self.user_word_embs(u_r_words.squeeze(2).long())  # B * N * d
            i_words = self.item_word_embs(i_r_words.squeeze(2).long())  # B * N * d
            p_u, p_i = w_coatt(u_words, i_words)                 # B * N * 1
            u_w_fea = u_words.permute(0, 2, 1).bmm(p_u).squeeze(2)
            i_w_fea = u_words.permute(0, 2, 1).bmm(p_i).squeeze(2)

            user_review_embedding = self.dropout(self.relu(self.user_fc_layer(u_w_fea)))  # [batch, review_hidden_size]
            item_review_embedding = self.dropout(self.relu(self.item_fc_layer(i_w_fea)))
            review_embedding = torch.stack((user_review_embedding, item_review_embedding), 1)
            # --------review_feature_co-attention --------------------
            review_att, fea_att = fea_coatt(review_embedding, feature_embedding)
            fea_att_embedding = feature_embedding * fea_att
            user_id_emb, item_id_emb, travel_type_emd, travel_month_emd = torch.split(fea_att_embedding, 1, 1)
            user_id_emb = self.dropout(torch.sum(user_id_emb, 1))  # [batch, review_hidden_size]
            item_id_emb = self.dropout(torch.sum(item_id_emb, 1))  # [batch, review_hidden_size]
            travel_type_emd = self.dropout(torch.sum(travel_type_emd, 1))  # [batch, review_hidden_size]
            travel_month_emd = self.dropout(torch.sum(travel_month_emd, 1))  # [batch, review_hidden_size]

            review_att_embedding = review_embedding * review_att
            user_2att_review_embedding, item_2att_review_embedding = torch.split(review_att_embedding,[1, 1],1)

            user_fea = user_2att_review_embedding.squeeze(1)  # [batch, review_hidden_size]
            item_fea = user_2att_review_embedding.squeeze(1)   # [batch, review_hidden_size]
            u_fea.append(user_fea)
            i_fea.append(item_fea)

        u_fea = torch.cat(u_fea, 1)
        i_fea = torch.cat(i_fea, 1)

        u_fea = self.dropout(self.u_fc(u_fea))
        i_fea = self.dropout(self.i_fc(i_fea))

        user_fea = torch.stack([user_id_emb, travel_type_emd, u_fea], 1)
        item_fea = torch.stack([item_id_emb, travel_month_emd, i_fea], 1)

        return user_fea, item_fea

    def review_gate(self, reviews):
        # Eq 1
        reviews = reviews.sum(2)
        return torch.sigmoid(self.fc_g1(reviews)) * torch.tanh(self.fc_g2(reviews))

    def reset_para(self):
        for fc in [self.fc_g1, self.fc_g2, self.user_fc_layer, self.item_fc_layer,self.u_fc,self.i_fc]:
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
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)


class Co_Attention(nn.Module):
    '''
    review-level and word-level co-attention module
    Eq (2,3, 10,11)
    '''
    def __init__(self, dim, gumbel, pooling):
        super(Co_Attention, self).__init__()
        self.gumbel = gumbel
        self.pooling = pooling
        self.M = nn.Parameter(torch.randn(dim, dim))
        self.fc_u = nn.Linear(dim, dim)
        self.fc_i = nn.Linear(dim, dim)

        self.reset_para()

    def reset_para(self):
        nn.init.xavier_uniform_(self.M, gain=1)
        nn.init.uniform_(self.fc_u.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_u.bias, -0.1, 0.1)
        nn.init.uniform_(self.fc_i.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc_i.bias, -0.1, 0.1)

    def forward(self, u_fea, i_fea):
        '''
        u_fea: B * L1 * d
        i_fea: B * L2 * d
        return:
        B * L1 * 1
        B * L2 * 1
        '''
        u = self.fc_u(u_fea)
        i = self.fc_i(i_fea)
        xx=u.matmul(self.M)
        S = u.matmul(self.M).bmm(i.permute(0, 2, 1))  # B * L1 * L2 Eq(2/10), we transport item instead user
        if self.pooling == 'max':
            u_score = S.max(2)[0]  # B * L1
            i_score = S.max(1)[0]  # B * L2
        else:
            u_score = S.mean(2)  # B * L1
            i_score = S.mean(1)  # B * L2
        if self.gumbel:
            p_u = F.gumbel_softmax(u_score, hard=True, dim=1)
            p_i = F.gumbel_softmax(i_score, hard=True, dim=1)
        else:
            p_u = F.softmax(u_score, dim=1)
            p_i = F.softmax(i_score, dim=1)
        return p_u.unsqueeze(2), p_i.unsqueeze(2)
