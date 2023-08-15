import torch
import pandas as pd
import os
from tqdm import tqdm
import random
import numpy as np
from config import args
from utils import *
from ERDNet import ERDNet
import dgl

import warnings
import json
import scipy.sparse as sp
import time

warnings.filterwarnings('ignore')


def train_eval(args=None, dataset='YAGO'):
    train_list, valid_list, test_list, train_times, valid_times, test_times, num_ent, num_rel, num_time \
        = dataloader_from_raw(dataset)  # load data
    train_g_list, valid_g_list, test_g_list = load_graph(dataset, device)  # load dgl graph, erinfo
    all_facts_filter = train_list+valid_list+test_list
    all_facts_filter = np.concatenate(all_facts_filter)
    all_facts_filter = add_inverse_rel(all_facts_filter, num_rel)  # inverse relations


    if args.freq_info:
        all_times = train_times + valid_times + test_times
        time_tail_freq_list = load_freq(dataset, all_times, num_ent, num_rel)  # history statistics info
    else:
        time_tail_freq_list = None
    print("====================All data loaded!====================")
    # return 0,0,0,0
    # -----------------------------------------load-----------------------------------------

    model = DRRNet(num_ent, num_rel, h_dim=args.n_hidden, n_layers=2, dropout=args.dropout,
                   input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
                   feature_map_dropout=args.feat_dropout, device=device, alpha=args.alpha)
    model.to(device)

    # ----------------------------------------test----------------------------------------
    if args.mode == 'test':
        print("----------------------------------------test----------------------------------------\n")
        model_state_file = '../results/bestmodel/model_{}_freqinfo{}_bias{}_alpha{}.pth'.format(dataset,
                                                                                                args.freq_info,
                                                                                                args.use_bias,
                                                                                                args.alpha)
        print("use model file to test: {}".format(model_state_file))
        mrr, hits1, hits3, hits10 = test(model, dataset, train_g_list + valid_g_list, test_g_list, test_list,
                                         time_tail_freq_list, num_ent, num_rel, mode='test', device=device,
                                         model_state_file=model_state_file, all_facts_filter=all_facts_filter)
        print("   mrr in test data:", mrr)
        print(" hits1 in test data:", hits1)
        print(" hits3 in test data:", hits3)
        print("hits10 in test data:", hits10)
        return mrr, hits1, hits3, hits10

    print("----------------------------------------start training----------------------------------------\n")
    best_mrr = 0
    if args.mode == 'pretrain':
        batchsize_dict = {}
        batchsize_dict['YAGO'] = 2048
        batchsize_dict['WIKI'] = 2048
        batchsize_dict['ICEWS14'] = 512
        batchsize_dict['ICEWS18'] = 2048
        batchsize_dict['GDELT'] = 2500

        model_state_file = '../results/bestmodel/model_{}_pretrain.pth'.format(dataset)
        # model_state_file = '../results/bestmodel/model_{}_pretrain_test_gap.pth'.format(dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_mrr = 0
        patience = 0  # early stop
        neg_ratio = 2  # num of neg sample

        batchsize = batchsize_dict.get(dataset)

        for epoch in range(args.max_epochs):
            model.train()
            losses = []
            train_data = np.concatenate(train_list)
            np.random.shuffle(train_data)
            train_data = add_inverse_rel(train_data, num_rel)  #
            train_data_set = set((item[0], item[1], item[2]) for item in train_data)

            for start_idx in tqdm(range(0, train_data.shape[0], batchsize)):
                end_idx = min(train_data.shape[0], start_idx + batchsize)
                batch_data = train_data[start_idx: end_idx]
                batch_data_neg = []  # 负采样
                for item in batch_data:
                    count = 0
                    while count < neg_ratio:
                        neg_o = np.random.randint(num_ent)
                        if (item[0], item[1], neg_o) not in train_data_set:
                            batch_data_neg.append([item[0], item[1], neg_o])
                            count += 1

                batch_data = torch.tensor(batch_data).long().to(device)
                batch_data_neg = torch.tensor(batch_data_neg).long().to(device)
                loss = model.loss_func(None, batch_data, batch_data_neg, neg_ratio)
                losses.append(loss.item())
                # print(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch {:04d} | Ave Loss: {:.4f}".format(epoch, np.mean(losses)))
            # validate
            if epoch and epoch % args.valid_step == 0:
                ranks = []
                valid_data = np.concatenate(valid_list)
                valid_data = add_inverse_rel(valid_data, num_rel)  #
                for start_idx in range(0, valid_data.shape[0], 512):
                    end_idx = min(valid_data.shape[0], start_idx + 512)
                    batch_data_ = valid_data[start_idx: end_idx]
                    batch_data_ = torch.tensor(batch_data_).long().to(device)
                    facts_output, scores_output = model.predict(None, num_rel, batch_data_)
                    ranks_tmp = get_ranks(facts_output, scores_output, filter=args.filter, num_ent=num_ent)
                    ranks.append(ranks_tmp)
                mrr, hits1, hits3, hits10 = stat_ranks(torch.cat(ranks))

                print("mrr in valid data: {:.4f}".format(mrr))
                if mrr < best_mrr:
                    patience += 1
                    if epoch >= args.max_epochs or patience > 3:  # early stop
                        break
                else:
                    best_mrr = mrr
                    print("!!! Best model update with history_len: {}   Best_mrr:{:.4f}".format(args.train_history_len,
                                                                                                best_mrr))
                    torch.save(
                        {'state_dict': model.state_dict(), 'epoch': epoch, 'history_len': args.train_history_len},
                        model_state_file)  #
        ranks = []
        test_data = np.concatenate(test_list)
        test_data = add_inverse_rel(test_data, num_rel)  #
        for start_idx in range(0, test_data.shape[0], 512):
            end_idx = min(test_data.shape[0], start_idx + 512)
            batch_data_ = test_data[start_idx: end_idx]
            batch_data_ = torch.tensor(batch_data_).long().to(device)
            facts_output, scores_output = model.predict(None, num_rel, batch_data_)
            ranks_tmp = get_ranks(facts_output, scores_output, filter=args.filter, num_ent=num_ent)
            ranks.append(ranks_tmp)
        mrr, hits1, hits3, hits10 = stat_ranks(torch.cat(ranks))
        print("-------Pretrain outcome on {} test data-------".format(dataset))
        print("   mrr in test data:", mrr)
        print(" hits1 in test data:", hits1)
        print(" hits3 in test data:", hits3)
        print("hits10 in test data:", hits10)
        return mrr, hits1, hits3, hits10

    elif args.mode == 'train':
        model_state_file = '../results/bestmodel/model_{}_freqinfo{}_bias{}_alpha{}.pth'.format(dataset,
                                                                                                args.freq_info,
                                                                                                args.use_bias,
                                                                                                args.alpha)

        model_state_file_prev = '../results/bestmodel/model_{}_pretrain.pth'.format(dataset)
        checkpoint = torch.load(model_state_file_prev, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

        # if dataset in ['YAGO', 'WIKI']:
        #     print("Freeze embedding and online training!...")
        #     model.ent_embs.requires_grad = False  # Freeze, or the outcome gets worse
        #     model.rel_embs.requires_grad = False
        print("Freeze")
        model.ent_embs.requires_grad = False  # Freeze, or the outcome gets worse
        model.rel_embs.requires_grad = False

        # num_params = 0
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if 'layers' in name:
        #             num_params += param.numel()/2
        #         else:
        #             num_params += param.numel()
        #         print(name)
        # print("model parmameters for {}: {}".format(args.dataset, num_params))
        # return 0, 0, 0, 0



        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2,
                                                               verbose=True,
                                                               threshold=0.01, cooldown=0, min_lr=0.00001, eps=1e-08)
        best_mrr = 0
        patience = 0  # early stop
        for epoch in range(1, args.max_epochs):
            model.train()
            index = [_ for _ in range(len(train_list))]  #
            random.shuffle(index)
            losses = []
            for T_idx in tqdm(index, ncols=100):
                # for T_idx in index:
                if T_idx == 0: continue  #
                # if T_idx - args.train_history_len < 0: continue  #
                output = np.asarray(train_list[T_idx])  #
                output_w_inverse = add_inverse_rel(output, num_rel)
                if args.freq_info:
                    all_tail_seq = time_tail_freq_list[T_idx]  #
                    seq_idx = output_w_inverse[:, 0] * num_rel * 2 + output_w_inverse[:, 1]  #
                    history_tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                    one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)
                    history_tail_seq.to(device)
                    one_hot_tail_seq.to(device)
                else:
                    history_tail_seq = None
                    one_hot_tail_seq = None
                history_glist = train_g_list[max(0, T_idx - args.train_history_len): T_idx]  #
                output_w_inverse = torch.tensor(output_w_inverse).long().to(device)  #

                loss = model.loss_func(history_glist, output_w_inverse, None, history_tail_seq, one_hot_tail_seq,
                                       args.use_bias, args.freq_info)  #
                losses.append(loss.item())
                # print("Timestamp:{}  Loss: {}： ".format(T_idx, loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # scheduler.step(np.mean(losses))  #
            print("Epoch {:04d} | Ave Loss: {:.4f}".format(epoch, np.mean(losses)))
            # 验证模型效果
            if epoch and epoch % args.valid_step == 0:
                mrr, hits1, hits3, hits10 = test(model, dataset, train_g_list, valid_g_list, valid_list,
                                                 time_tail_freq_list, num_ent, num_rel,
                                                 mode='train', device=device, history_len=args.train_history_len)
                print("mrr in valid data: {:.4f}".format(mrr))
                if mrr < best_mrr:
                    patience += 1
                    if epoch >= args.max_epochs or patience > 3:
                        break
                else:
                    best_mrr = mrr
                    print("!!! Best model update with history_len {}".format(args.train_history_len))
                    torch.save(
                        {'state_dict': model.state_dict(), 'epoch': epoch, 'history_len': args.train_history_len},
                        model_state_file)  # 存储最优模型
        print("Best raw model: {}  best mrr in valid data:{:.4f}".format(model_state_file, best_mrr))
        mrr, hits1, hits3, hits10 = test(model, dataset, train_g_list + valid_g_list, test_g_list, test_list,
                                         time_tail_freq_list, num_ent, num_rel,
                                         mode='test', device=device, model_state_file=model_state_file)
        print("   mrr in test data:", mrr)
        print(" hits1 in test data:", hits1)
        print(" hits3 in test data:", hits3)
        print("hits10 in test data:", hits10)

        return mrr, hits1, hits3, hits10

def test(model, dataset, history_g_list, test_g_list, test_list, time_tail_freq_list, num_ent, num_rel, mode,
         device='cpu', model_state_file=None, history_len=2, all_facts_filter=None):
    with torch.no_grad():
        if mode == "test":
            checkpoint = torch.load(model_state_file)
            model.load_state_dict(checkpoint['state_dict'])  #
            test_history_len = checkpoint['history_len']
            print("Best model from epoch {}, with history length of {}".format(checkpoint['epoch'], test_history_len))
        else:
            test_history_len = history_len

        model.eval()
        ranks = []

        for T_idx, test_snap in enumerate(tqdm(test_list, ncols=50)):
            if T_idx - test_history_len < 0:
                history_glist = [g for g in history_g_list[T_idx - test_history_len:]] \
                                + [g for g in test_g_list[0: T_idx]]
            else:
                history_glist = [g for g in test_g_list[T_idx - test_history_len: T_idx]]
            output_w_inverse = add_inverse_rel(test_snap, num_rel)
            output_w_inverse = torch.LongTensor(output_w_inverse)
            if args.freq_info:
                all_tail_seq = time_tail_freq_list[len(history_g_list) + T_idx]
                seq_idx = output_w_inverse[:, 0] * num_rel * 2 + output_w_inverse[:, 1]  # (h,r ,?)
                history_tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())  #
                one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)
                # history_tail_seq.to(device)
                # one_hot_tail_seq.to(device)
                # all_static_tail_seq = time_tail_freq_list[-1]
                # one_hot_tail_seq = history_tail_seq.masked_fill(torch.tensor(all_static_tail_seq[seq_idx].todense()) != 0, 1)
            else:
                history_tail_seq = None
                one_hot_tail_seq = None

            facts_output, scores_output = model.predict(history_glist, num_rel, output_w_inverse.to(device),
                                                        history_tail_seq,
                                                        one_hot_tail_seq,
                                                        args.freq_info, args.use_bias)
            ranks_tmp = get_ranks(facts_output, scores_output, filter=args.filter, num_ent=num_ent, all_facts_filter=all_facts_filter)
            ranks.append(ranks_tmp)
        mrr, hits1, hits3, hits10 = stat_ranks(torch.cat(ranks))
    return mrr, hits1, hits3, hits10


if __name__ == '__main__':
    seed = 40
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:{}'.format(args.device_id))
    dataset = args.dataset

    print(args)

    mrr_list = []
    hits1_list = []
    hits3_list = []
    hits10_list = []

    for round in range(1):
        mrr, hits1, hits3, hits10 = train_eval(args, dataset)  #

        mrr_list.append(mrr)
        hits1_list.append(hits1)
        hits3_list.append(hits3)
        hits10_list.append(hits10)

    print(mrr_list)
    print(hits1_list)
    print(hits3_list)
    print(hits10_list)

    mrr_mean = np.mean(mrr_list)
    mrr_var = np.var(mrr_list)
    mrr_mean = ('%.4f' % mrr_mean)
    mrr_var = ('%.4f' % mrr_var)

    hits1_mean = np.mean(hits1_list)
    hits1_var = np.var(hits1_list)
    hits1_mean = ('%.4f' % hits1_mean)
    hits1_var = ('%.4f' % hits1_var)

    hits3_mean = np.mean(hits3_list)
    hits3_var = np.var(hits3_list)
    hits3_mean = ('%.4f' % hits3_mean)
    hits3_var = ('%.4f' % hits3_var)

    hits10_mean = np.mean(hits10_list)
    hits10_var = np.var(hits10_list)
    hits10_mean = ('%.4f' % hits10_mean)
    hits10_var = ('%.4f' % hits10_var)

    res_path = '../eva_result_final.csv'
    exp_result = []
    model_name = 'model_{}_freqinfo{}_verse={}_alpha={}.pth'.format(dataset, args.freq_info, args.use_bias, args.alpha)
    heads = ['dataset', 'dim', 'lr', 'freq_info', 'use_bias', 'alpha', 'mrr', 'hits1', 'hits3', 'hits10', 'mode',
             'history_len']
    exp_result.append(
        (dataset, args.n_hidden, args.lr, args.freq_info, args.use_bias, args.alpha,
         "{}+-{}".format(mrr_mean, mrr_var),
         "{}+-{}".format(hits1_mean, hits1_var),
         "{}+-{}".format(hits3_mean, hits3_var),
         "{}+-{}".format(hits10_mean, hits10_var),
         args.mode, args.train_history_len))

    # exp_result.append(
    #     (dataset, args.n_hidden, args.lr, args.freq_info, args.use_bias, args.alpha,
    #      "{}".format(mrr_mean),
    #      "{}".format(hits1_mean),
    #      "{}".format(hits3_mean),
    #      "{}".format(hits10_mean),
    #      'non-freeze', args.train_history_len))

    # exp_result = pd.DataFrame(exp_result, columns=heads)
    # if os.path.exists(res_path):
    #     exp_result.to_csv(res_path, header=0, index=False, mode='a')
    # else:
    #     exp_result.to_csv(res_path, index=False)
