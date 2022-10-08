# -*- encoding: utf-8 -*-
import time
import random
import math
import fire

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from dataset import ReviewData
from framework import Model
import models
import config
import datetime
import os
import threading
def now():
    return datetime.datetime.now()


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def train(**kwargs):
    opt = getattr(config, 'DefaultConfig')()
    opt.parse(dataset,kwargs)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f'train data: {len(train_data)}; test data: {len(val_data)}')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # training
    print("start training....")
    min_loss = 1e+10
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()

    for epoch in range(opt.num_epochs):
        epoch_start_times=now()
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        print(f"{now()}  Epoch {epoch}...")
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            train_datas = unpack_input(opt, train_datas)

            optimizer.zero_grad()
            output = model(train_datas)
            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()
            smooth_mae_loss = smooth_mae_func(output, scores)

            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss
            loss.backward()
            optimizer.step()
            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    print("\t{}, {} step finised;".format(now(), idx))
                    val_loss, val_mse, val_mae, val_rmse = predict(model, val_data_loader, opt)

                    if val_loss < min_loss:
                        model.save(opt.pth_path)
                        min_loss = val_loss
                        print("\tmodel save")
                    if val_loss > min_loss:
                        best_res = min_loss

        scheduler.step()
        mse = total_loss * 1.0 / len(train_data)
        print(f"\ttrain data: loss:{total_loss:.4f}, mse: {mse:.4f};")

        val_loss, val_mse, val_mae, val_rmse = predict(model, val_data_loader, opt)
        if val_loss < min_loss:
            model.save(opt.pth_path)
            min_loss = val_loss
            # print("model save")
        if val_mse < best_res:
            best_res = val_mse
        # print("*"*30)
    print(f"{opt.model} {dataset} {opt.print_opt} best_res:{best_res}")
    print("----" * 20)

    return best_res

def test(**kwargs):
    print('test_1')
    opt = getattr(config, 'DefaultConfig')()
    opt.parse(dataset, kwargs)

    assert(len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    # print(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    # print(f"{now()}: test in the test datset")

    predict_loss, test_mse, test_mae,test_rmse = predict(model, test_data_loader, opt)
    print(f"{opt.model}_evaluation reslut: test_mse: {test_mse:.4f}; test_mae: {test_mae:.4f};test_rmse: {test_rmse:.4f}")
    return predict_loss, test_mse, test_mae, test_rmse


def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0
    total_rmseloss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            test_data = unpack_input(opt, test_data)

            output = model(test_data)

            mse_loss = torch.sum((output-scores)**2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()

            rmse_loss = torch.sqrt(mse_loss)
            total_rmseloss+= rmse_loss.item()


    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    rmse = total_rmseloss * 1.0 / data_len
    model.train()
    return total_loss, mse, mae,rmse


def unpack_input(opt, x):

    uids, iids,travel_type,travel_month = list(zip(*x))
    uids = list(uids)
    iids = list(iids)
    travel_type = list(travel_type)
    travel_month= list(travel_month)

    user_reviews = opt.users_review_list[uids]
    user_item2id = opt.user2itemid_list[uids]  # 检索出该user对应的item id

    user_doc = opt.user_doc[uids]

    item_reviews = opt.items_review_list[iids]
    item_user2id = opt.item2userid_list[iids]  # 检索出该item对应的user id
    item_doc = opt.item_doc[iids]

    data = [user_reviews, item_reviews, uids, iids,user_item2id, item_user2id, user_doc,item_doc,travel_type,travel_month,]


    data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
    return data

def model_run(dataset,model_name,gpu_id,num_fea,output,batch=128,**kwargs):
    opt = getattr(config, 'DefaultConfig')()
    opt.parse(dataset, kwargs)
    print('model:', model_name)
    pth_path = f'./checkpoints/{model_name}_{dataset}.pth'
    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints/')

    best_res = train(dataset = dataset, model=model_name, pth_path=pth_path,num_fea=num_fea, output=output, gpu_id=gpu_id)
    predict_loss, test_mse, test_mae,test_rmse = test(dataset = dataset, model=model_name, pth_path=pth_path, num_fea=num_fea, output=output,gpu_id=gpu_id)
    return [dataset,model_name,test_mse,test_mae,test_rmse]


"""重新定义带返回值的线程类"""
class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


if __name__ == "__main__":
    for dataset in ['weight_5']:
        print(dataset)
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('time',time)
        final_df = pd.DataFrame(columns =['dataset','model','MSE','MAE','RMSE'])
        model = 'HCOATT_NO_CNN'
        # final_df.loc[len(final_df)] = model_run(dataset, 'MPCN', 2, 1, 'fm')
        # final_df.loc[len(final_df)] = model_run(dataset, 'NARRE', 2, 2, 'lfm')
        # # final_df.loc[len(final_df)] = model_run(dataset, 'D_ATTN', 2, 1, 'fm')
        # final_df.loc[len(final_df)] = model_run(dataset, 'HANCI', 2, 2, 'lfm')
        # final_df.loc[len(final_df)] = model_run(dataset, 'DeepCoNN', 2,1,'fm')
        final_df.loc[len(final_df)] = model_run(dataset, 'HCOATT_NO_CNN', 3, 3, 'fm')
        final_df.to_excel(f'results/final_df_{dataset}_{model}.xlsx', index=False)
