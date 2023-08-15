import collections
import random
import time

import torch
import torch.nn as nn
import dgl
import numpy as np
from collections import defaultdict
import pandas as pd
import json
import os
import scipy.sparse as sp
from tqdm import tqdm


def build_sub_graph(num_nodes, num_rels, num_types, facts, device):  #
    '''
    :param num_nodes:
    :param num_rels:
    :param triples:  [[s, r, o], [s, r, o], ...]
    :param use_cuda:
    :param gpu:
    :return:
    '''

    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        out_deg = g.out_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        out_deg[torch.nonzero(out_deg == 0).view(-1)] = 1
        in_norm = 1.0 / in_deg
        out_norm = 1.0 / out_deg
        return in_norm, out_norm

    def get_typeid(src, src_type):
        df = pd.DataFrame()
        df['ent'] = src
        df['ent_type'] = src_type
        df = df.sort_values(by=['ent']).drop_duplicates()
        type_id = df['ent_type'].tolist()
        return np.array(type_id)

    # inverse_facts = facts[:, [2, 1, 0, 3, 5, 4]]
    inverse_facts = facts[:, [2, 1, 0, 3]]
    inverse_facts[:, 1] = inverse_facts[:, 1] + num_rels
    facts = np.concatenate([facts, inverse_facts])
    # src, rel, dst, _, src_type, dst_type = facts.transpose()
    src, rel, dst, _ = facts.transpose()

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)

    in_norm, out_norm = comp_deg_norm(g)

    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)  # [0, num_nodes)
    # type_id = get_typeid(src, src_type)
    # type_id = torch.from_numpy(type_id).long().view(-1, 1)
    g.ndata.update(
        {'id': node_id, 'out_norm': out_norm.view(-1, 1), 'in_norm': in_norm.view(-1, 1)})  # (num_nodes, 1)

    g.apply_edges(lambda edges: {'up_norm': edges.src['in_norm']})
    g.edata['type'] = torch.LongTensor(rel)


    # uniq_e, e_to_r, uniq_r, r_to_r = e_r_info(facts, num_nodes,2*num_rels)
    # g.uniq_e = uniq_e
    # g.uniq_r = uniq_r
    # g.e_to_r = e_to_r
    # g.r_to_r = r_to_r
    g = g.to(device)

    return g


def e_r_info(facts, num_rels):  # triplets: [[s, r, o], [s, r, o], ...]
    # inverse_facts = facts[:, [2, 1, 0, 3, 5, 4]]
    inverse_facts = facts[:, [2, 1, 0, 3]]
    inverse_facts[:, 1] = inverse_facts[:, 1] + num_rels
    facts = np.concatenate([facts, inverse_facts])
    # src, rel, dst, _, src_type, dst_type = facts.transpose()

    src, rel, dst, _ = facts.transpose()  #  (sub, rel, obj, timestamp, sub_type, obj_type)
    uniq_e = np.unique(src)
    uniq_r = np.unique(rel)
    e_to_r = defaultdict(set)
    r_to_r = defaultdict(set)
    r_to_e = defaultdict(set)
    df_now = pd.DataFrame(facts)
    for idx in range(df_now.shape[0]):
        item = df_now.iloc[idx]
        h = item[0]
        r = item[1]
        t = item[2]
        co_occur_r = set(df_now[(df_now[0] == h) & (df_now[1] != r)][1].tolist())
        for r_ in co_occur_r:
            r_to_r[r].add(r_)
        e_to_r[h].add(r)
        r_to_e[r].add(h)
        r_to_e[r].add(t)

    r_to_r_list = []
    for r in uniq_r:
        r_to_r[r].add(int(r))
        tmp_list = np.array(list(r_to_r[r])).tolist()
        r_to_r_list.append(tmp_list)

    r_to_e_list = []
    for r in uniq_r:
        tmp_list = np.array(list(r_to_e[r])).tolist()
        # print(type(tmp_list[0]))
        # tmp_list = np.array(r_to_e[r]).tolist()
        r_to_e_list.append(tmp_list)
    info_dict = {}
    # info_dict['uniq_e'] = uniq_e.tolist()
    # info_dict['e_to_r'] = e_to_r_list
    info_dict['uniq_r'] = uniq_r.tolist()
    info_dict['r_to_r'] = r_to_r_list
    info_dict['r_to_e'] = r_to_e_list

    return info_dict


def get_ranks(test_facts, scores, batch_size=1000, filter=0, num_ent=0, all_facts_filter=None):
    ranks = []
    if filter:
        all_facts = test_facts.to('cpu').numpy()
        all_facts_set = set((item[0], item[1], item[2]) for item in all_facts)

        if all_facts_filter is not None:
            # all_facts_filter = np.array(all_facts_filter)
            all_facts_set = set((item[0], item[1], item[2]) for item in all_facts_filter)
            # print("filtered all")

        for idx in range(len(all_facts)):
            tmp_fact = all_facts[idx]
            score = scores[idx]
            filter_t = []
            # print((tmp_fact[0], tmp_fact[1], tmp_fact[2]) in all_facts_set)
            for t_ in range(num_ent):
                if (tmp_fact[0], tmp_fact[1], t_) in all_facts_set:
                    filter_t.append(t_)
            filter_t.remove(tmp_fact[2])
            score[np.asarray(filter_t)] = -np.inf
            _, indices = torch.sort(score, descending=True)
            indice = torch.nonzero(indices == tmp_fact[2]).item()
            ranks.append(indice)
        ranks = torch.tensor(ranks)
        ranks += 1
    else:
        num_facts = len(test_facts)
        n_batch = (num_facts + batch_size - 1) // batch_size
        ranks = []
        for idx in range(n_batch):
            batch_start = idx * batch_size
            batch_end = min(num_facts, (idx + 1) * batch_size)
            batch_facts = test_facts[batch_start:batch_end, :]
            batch_scores = scores[batch_start:batch_end, :]
            target = test_facts[batch_start:batch_end, 2]
            ranks.append(sort_rank(batch_scores, target))
        ranks = torch.cat(ranks)
        ranks += 1
    return ranks


def sort_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def stat_ranks(ranks):
    mrr = torch.mean(1.0 / ranks.float())
    hits1 = torch.mean((ranks <= 1).float())
    hits3 = torch.mean((ranks <= 3).float())
    hits10 = torch.mean((ranks <= 10).float())
    return mrr.item(), hits1.item(), hits3.item(), hits10.item()


def add_inverse_rel(data, num_rel):  # [[s, r, o], [], ...]
    inverse_triples = np.array([[o, r + num_rel, s, t] for s, r, o, t in data])
    triples_w_inverse = np.concatenate((data, inverse_triples))
    return triples_w_inverse


#####################################Load data#####################################
def load_quadruples(in_Path, file_name):
    with open(os.path.join(in_Path, file_name), 'r') as f_read:
        quadruple_list = []
        timestamp_all = set()
        for line in f_read:
            line_splitted = line.split()
            head = int(line_splitted[0])
            relation = int(line_splitted[1])
            tail = int(line_splitted[2])
            timestamp = int(line_splitted[3])
            # quadruple_list.append((h, r, t, timestamp))
            quadruple_list.append((head, relation, tail, timestamp))
            timestamp_all.add(timestamp)
    timestamp_all = list(timestamp_all)
    timestamp_all.sort()
    return np.asarray(quadruple_list), timestamp_all


def dataloader_from_raw(dataset):
    print("==============load dataloader from raw files")

    dataset = dataset.split('_')[0] # ICEWS14_delete10

    data_path = '../data/{}'.format(dataset)
    # data_path = 'data/{}'.format(dataset)
    train_data, train_times = load_quadruples(data_path, 'train.txt')
    valid_data, valid_times = load_quadruples(data_path, 'valid.txt')
    test_data, test_times = load_quadruples(data_path, 'test.txt')
    data_all = np.concatenate([train_data, valid_data, test_data])

    df_all = pd.DataFrame(data_all)
    num_ent = max(set(df_all[0].tolist() + df_all[2].tolist())) + 1
    num_rel = max(set(df_all[1].tolist())) + 1
    num_time = len(set(df_all[3].tolist())) + 1

    train_list = []
    valid_list = []
    test_list = []

    for T in train_times:
        facts = df_all[df_all[3] == T].values.tolist()
        train_list.append(facts)
    for T in valid_times:
        facts = df_all[df_all[3] == T].values.tolist()
        valid_list.append(facts)
    for T in test_times:
        facts = df_all[df_all[3] == T].values.tolist()
        test_list.append(facts)

    # for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    #     ratio = 0.2
    #     train_list = []
    #     valid_list = []
    #     test_list = []
    #     num_ent_delete = random.sample(range(0,num_ent), int(num_ent*ratio))
    #     print("left ents: ",len(num_ent_delete))
    #     all_trainsamples = 0
    #     for T in train_times:
    #         facts = df_all[(df_all[3] == T) & (df_all[0].isin(num_ent_delete))& (df_all[2].isin(num_ent_delete))].values.tolist()
    #         train_list.append(facts)
    #         all_trainsamples +=len(facts)
    #     print("left train samples: ", all_trainsamples)
    #     # if(all_trainsamples/len(num_ent_delete)>30):
    #     if(1):
    #         print("find: ", len(num_ent_delete))
    #         for T in valid_times:
    #             facts = df_all[(df_all[3] == T) & (df_all[0].isin(num_ent_delete))& (df_all[2].isin(num_ent_delete))].values.tolist()
    #             valid_list.append(facts)
    #         for T in test_times:
    #             facts = df_all[(df_all[3] == T) & (df_all[0].isin(num_ent_delete))& (df_all[2].isin(num_ent_delete))].values.tolist()
    #             test_list.append(facts)
    #         break
    #     else: continue
    return train_list, valid_list, test_list, train_times, valid_times, test_times, num_ent, num_rel, num_time


def load_freq(dataset, all_times, num_ent, num_rel):
    print("==============load freq info")
    freq_path = '../data_cached/'
    if not os.path.exists(freq_path):
        os.makedirs(freq_path)
    time_tail_seq_list = []
    time_tail_freq_list = []
    timelast = time.time()
    for idx in tqdm(range(len(all_times)), ncols=50):
        time_tail_seq = sp.load_npz('../data_cached/{}/history_seq/h_r_history_seq_{}.npz'.format(
            dataset, idx))
        time_tail_seq_list.append(time_tail_seq)

    for idx in tqdm(range(len(all_times)), ncols=50):
        if idx == 0:
            time_tail_freq = sp.csr_matrix(([], ([], [])), shape=(num_ent * (num_rel * 2), num_ent))
            time_tail_freq_list.append(time_tail_freq)
            continue
        else:
            time_tail_freq = time_tail_seq_list[idx - 1] + time_tail_freq_list[idx - 1]
            time_tail_freq_list.append(time_tail_freq)
        #
        # if len(time_tail_freq_list)<4:
        #     for item in time_tail_seq_list[:-1]:
        #         time_tail_freq += item
        # else:
        #     for item in time_tail_seq_list[-4:-1]:
        #         time_tail_freq +=item
        #
        # time_tail_freq+=time_tail_freq_list[-1]
    return time_tail_freq_list


def load_graph(dataset, device):
    print("==============load graphs and json files")
    dgl_path = '../data_cached/{}'.format(dataset) + '/dgl_graph/{}_{}'
    json_path = '../data_cached/{}'.format(dataset) + '/e_r_info_json/{}_{}'
    # if (0):
    if (os.path.exists(dgl_path.format(dataset, 'train.dgl'))):
        train_g_list, _ = dgl.load_graphs(dgl_path.format(dataset, 'train.dgl'))
        valid_g_list, _ = dgl.load_graphs(dgl_path.format(dataset, 'valid.dgl'))
        test_g_list, _ = dgl.load_graphs(dgl_path.format(dataset, 'test.dgl'))
    else:
        print("===============graphs and e_r_info files not exist, process...")
        if not (os.path.exists('../data_cached/{}/dgl_graph'.format(dataset))):
            # os.mkdir('../data_cached/{}/dgl_graph'.format(dataset))
            # os.mkdir('../data_cached/{}/e_r_info_json'.format(dataset))
            os.makedirs('../data_cached/{}/dgl_graph'.format(dataset))
            os.makedirs('../data_cached/{}/e_r_info_json'.format(dataset))
            print("build new dir!")
        train_list, valid_list, test_list, train_times, valid_times, test_times, num_ent, num_rel, num_time = dataloader_from_raw(
            dataset)
        train_g_list = [build_sub_graph(num_ent, num_rel, None, np.asarray(snap), device) for snap in train_list]
        valid_g_list = [build_sub_graph(num_ent, num_rel, None, np.asarray(snap), device) for snap in valid_list]
        test_g_list = [build_sub_graph(num_ent, num_rel, None, np.asarray(snap), device) for snap in test_list]
        dgl.save_graphs(dgl_path.format(dataset, 'train.dgl'), train_g_list)
        dgl.save_graphs(dgl_path.format(dataset, 'valid.dgl'), valid_g_list)
        dgl.save_graphs(dgl_path.format(dataset, 'test.dgl'), test_g_list)
        train_info_dict = {}
        for i, snap in enumerate(train_list):
            train_info_dict[i] = e_r_info(np.asarray(snap), num_rel)
        with open(json_path.format(dataset, 'train.json'), 'w') as f:
            json.dump(train_info_dict, f)
        valid_info_dict = {}
        for i, snap in enumerate(valid_list):
            valid_info_dict[i] = e_r_info(np.asarray(snap), num_rel)
        with open(json_path.format(dataset, 'valid.json'), 'w') as f:
            json.dump(valid_info_dict, f)
        test_info_dict = {}
        for i, snap in enumerate(test_list):
            test_info_dict[i] = e_r_info(np.asarray(snap), num_rel)
        with open(json_path.format(dataset, 'test.json'), 'w') as f:
            json.dump(test_info_dict, f)
        print("===============graphs and e_r_info files: Done!")

    train_g_list = [g.to(device) for g in train_g_list]
    valid_g_list = [g.to(device) for g in valid_g_list]
    test_g_list = [g.to(device) for g in test_g_list]

    with open(json_path.format(dataset, 'train.json'), 'r') as f:
        info_dict = json.load(f)
        for i in range(len(train_g_list)):
            item = info_dict['{}'.format(i)]
            # train_g_list[i].uniq_e = item['uniq_e']
            train_g_list[i].uniq_r = item['uniq_r']
            # train_g_list[i].e_to_r = item['e_to_r']
            train_g_list[i].r_to_r = item['r_to_r']
            train_g_list[i].r_to_e = item['r_to_e']
    with open(json_path.format(dataset, 'valid.json'), 'r') as f:
        info_dict = json.load(f)
        for i in range(len(valid_g_list)):
            item = info_dict['{}'.format(i)]
            # valid_g_list[i].uniq_e = item['uniq_e']
            valid_g_list[i].uniq_r = item['uniq_r']
            # valid_g_list[i].e_to_r = item['e_to_r']
            valid_g_list[i].r_to_r = item['r_to_r']
            valid_g_list[i].r_to_e = item['r_to_e']

    with open(json_path.format(dataset, 'test.json'), 'r') as f:
        info_dict = json.load(f)
        for i in range(len(test_g_list)):
            item = info_dict['{}'.format(i)]
            # test_g_list[i].uniq_e = item['uniq_e']
            test_g_list[i].uniq_r = item['uniq_r']
            # test_g_list[i].e_to_r = item['e_to_r']
            test_g_list[i].r_to_r = item['r_to_r']
            test_g_list[i].r_to_e = item['r_to_e']

    return train_g_list, valid_g_list, test_g_list
    # return train_list, valid_list, test_list, train_times, valid_times, test_times, num_ent, num_rel, num_time, train_g_list, valid_g_list, test_g_list
