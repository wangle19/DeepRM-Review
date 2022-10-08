# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NRPA(nn.Module):
    def __init__(self, opt):
        # :param review_size: review句子的数量
        # :param word_vec_dim: 词向量维度
        # :param conv_length: 卷积核的长度
        # :param conv_kernel_num: 卷积核数量
        # :param word_weights: 词向量矩阵权重
        # :param u_id_matrix_len: user_id总个数
        # :param i_id_matrix_len: item_id总个数
        # :param id_embedding_dim: id向量维度
        # :param atten_vec_dim: attention向量的维度
        super(NRPA, self).__init__()
        self.opt = opt
        self.review_size = opt.r_max_len
        w2v = torch.from_numpy(np.load(self.opt.w2v_path))
        self.word_weights = w2v
        self.user_reveiw_net = ReviewEncoder(
            word_vec_dim=opt.word_dim,
            conv_length=3,
            conv_kernel_num=32,
            word_weights=self.word_weights,
            id_matrix_len=opt.user_num,
            id_embedding_dim=32,
            atten_vec_dim=16
        )
        self.item_review_net = ReviewEncoder(
            word_vec_dim=opt.word_dim,
            conv_length=3,
            conv_kernel_num=32,
            word_weights=self.word_weights,
            id_matrix_len=opt.item_num,
            id_embedding_dim=32,
            atten_vec_dim=16
        )
        self.user_net = UIEncoder(
            conv_kernel_num=32,
            id_matrix_len=opt.user_num,
            id_embedding_dim=32,
            atten_vec_dim=16
        )
        self.item_net = UIEncoder(
            conv_kernel_num=32,
            id_matrix_len=opt.item_num,
            id_embedding_dim=32,
            atten_vec_dim=16
        )

    def forward(self, datas):
        user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc, travel_type, travel_month = datas
        # u_text (u_batch, review_size, review_length)
        # i_text (i_batch, review_size, review_length)
        batch_size      = len(uids)
        new_batch       = batch_size * self.review_size
        u_text          = user_reviews.reshape(new_batch, -1)
        mul_u_ids       = uids.unsqueeze(1)
        mul_u_ids       = torch.cat((mul_u_ids,) * self.review_size,dim=1).reshape(-1)
        d_matrix_user   = self.user_reveiw_net(u_text, mul_u_ids)
        d_matrix_user   = d_matrix_user.reshape(batch_size, self.review_size, -1).permute(0,2,1)

        i_text          = item_reviews.reshape(new_batch, -1)
        mul_i_ids       = iids.unsqueeze(1)
        mul_i_ids       = torch.cat((mul_i_ids,) * self.review_size,dim=1).reshape(-1)
        d_matrix_item   = self.item_review_net(i_text, mul_i_ids)
        d_matrix_item   = d_matrix_item.reshape(batch_size, self.review_size, -1).permute(0,2,1)

        u_fea = self.user_net(d_matrix_user, uids).squeeze(2)
        i_fea = self.item_net(d_matrix_item, iids).squeeze(2)
        return u_fea, i_fea


class ReviewEncoder(nn.Module):
    def __init__(self, word_vec_dim, conv_length, conv_kernel_num, word_weights,
                 id_matrix_len, id_embedding_dim, atten_vec_dim):
        # :param word_vec_dim: 词向量维度
        # :param conv_length: 卷积核的长度
        # :param conv_kernel_num: 卷积核数量
        # :param word_weights: 词向量矩阵权重
        # :param id_matrix_len: id总个数
        # :param id_embedding_dim: id向量维度
        # :param atten_vec_dim: attention向量的维度
        super(ReviewEncoder, self).__init__()
        self.embedding_review = nn.Embedding.from_pretrained(word_weights)
        self.embedding_id     = nn.Embedding(id_matrix_len, id_embedding_dim)
        self.conv = nn.Conv1d( # input shape (batch_size, review_length, word_vec_dim)
            in_channels = word_vec_dim,
            out_channels = conv_kernel_num,
            kernel_size = conv_length,
            padding = (conv_length -1) //2
        )# output shape (batch_size, conv_kernel_num, review_length)
        self.drop = nn.Dropout(p=1.0)
        self.l1 = nn.Linear(id_embedding_dim, atten_vec_dim)
        self.A1 = nn.Parameter(torch.randn(atten_vec_dim, conv_kernel_num), requires_grad=True)

    def forward(self, review, ids):
        # now the batch_size = user_batch * review_size
        review_vec      = self.embedding_review(review) #(batch_size, review_length, word_vec_dim)
        review_vec      = review_vec.permute(0, 2, 1)
        c               = F.relu(self.conv(review_vec)) #(batch_size, conv_kernel_num, review_length)
        c               = self.drop(c)
        id_vec          = self.embedding_id(ids) #(batch_size, id_embedding_dim)
        qw              = F.relu(self.l1(id_vec)) #(batch_size, atten_vec_dim)
        g               = torch.mm(qw, self.A1).unsqueeze(1) #(batch_size, 1, conv_kernel_num)
        g               = torch.bmm(g, c) #(batch_size, 1, review_length)
        alph            = F.softmax(g, dim=2) #(batch_size, 1, review_length)
        d               = torch.bmm(c, alph.permute(0, 2, 1)) #(batch_size, conv_kernel_num, 1)
        return d


class UIEncoder(nn.Module):
    def __init__(self, conv_kernel_num, id_matrix_len, id_embedding_dim, atten_vec_dim):

        # :param conv_kernel_num: 卷积核数量
        # :param id_matrix_len: id总个数
        # :param id_embedding_dim: id向量维度
        # :param atten_vec_dim: attention向量的维度
        super(UIEncoder, self).__init__()
        self.embedding_id = nn.Embedding(id_matrix_len, id_embedding_dim)
        self.review_f = conv_kernel_num
        self.l1 = nn.Linear(id_embedding_dim, atten_vec_dim)
        self.A1 = nn.Parameter(torch.randn(atten_vec_dim, conv_kernel_num), requires_grad=True)

    def forward(self, word_Att, ids):
        # now the batch_size = user_batch
        # word_Att => #(batch_size, conv_kernel_num, review_size)
        id_vec          = self.embedding_id(ids) #(batch_size, id_embedding_dim)
        qr              = F.relu(self.l1(id_vec)) #(batch_size, atten_vec_dim)
        e               = torch.mm(qr, self.A1).unsqueeze(1) #(batch_size, 1, conv_kernel_num)
        e               = torch.bmm(e, word_Att) #(batch_size, 1, review_size)
        beta            = F.softmax(e, dim=2) #(batch_size, 1, review_size)
        q               = torch.bmm(word_Att, beta.permute(0, 2, 1)) #(batch_size, conv_kernel_num, 1)
        return q