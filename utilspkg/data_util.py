##
# 这个文件用于数据的预处理，包括：数据格式处理、训练集和测试集数据加载、负采样方法
# 为了熟悉稳当操作，全部处理成dataframe格式#


import os
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from src.utils import dataloader_from_raw

cur_path = os.path.abspath('data_util.py')
root_path = cur_path[:cur_path.find('tkgc_mymodel')] + 'tkgc_mymodel'


# inpath = os.path.join(root_path, 'data/{}'.format(dataset))

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


#
def get_statistics(dataset):
    print("###############get dataset static: {}################".format(dataset))
    df = pd.read_csv(os.path.join(root_path, 'data_cached_new/{}/df_all_data.csv'.format(dataset)), header=0)
    # df1 = pd.read_csv(os.path.join(root_path, 'data_cached_new/{}/df_train_data_pos.csv'.format(dataset)), names=['h', 'r', 't', 'timestamp','h_type','t_type'])
    # timestamp_all = set(df['Time'].tolist())
    num_E = max(set(df['h'].tolist()) | set(df['t'].tolist())) + 1
    num_R = max(set(df['r'].tolist())) + 1
    type_entity = type_acquisition(dataset)  # entity: type
    # print(max(type_entity.values()))
    num_type = max(type_entity.values()) + 1
    print("num_E:{}, num_R:{}, num_type:{}\n".format(num_E, num_R, num_type))
    return num_E, num_R, num_type


#
def type_acquisition(dataset):
    # print("###############type acquisition: {}################".format(dataset))
    f_list = ['train.txt', 'test.txt', 'valid.txt']
    if dataset == 'ICEWS14':
        f_list = ['train.txt', 'test.txt']
    # quadruple_list = []
    triple_list = []
    # timestamp_all = set()
    entity_all = set()
    relation_all = set()
    for f in f_list:
        f_path = os.path.join(root_path, 'data/{}/{}'.format(dataset, f))
        with open(f_path, 'rb') as f_read:
            for line in f_read:
                line_splitted = line.split()
                head = int(line_splitted[0])
                relation = int(line_splitted[1])
                tail = int(line_splitted[2])
                # timestamp = int(line_splitted[3])
                # quadruple_list.append((head, relation, tail, timestamp))
                triple_list.append((head, relation, tail))
                entity_all.add(head)
                entity_all.add(tail)
                relation_all.add(relation)
                # timestamp_all.add(timestamp)
    sig = {}  # dict
    L = []
    type_entity = {}  # dict
    for e in entity_all:
        sig[e] = set()
    for triple in triple_list:
        h = triple[0]
        r = triple[1]
        t = triple[2]
        if ('head_of', r) not in sig.get(h):
            sig.get(h).add(('head_of', r))
        if ('tail_of', r) not in sig.get(t):
            sig.get(t).add(('tail_of', r))
    for e in entity_all:
        if sig.get(e) not in L:
            L.append(sig.get(e))
        # type.append((e, L.index(sig.get(e))))
        type_entity[e] = L.index(sig.get(e))
    # sig = {}
    # L = []
    # for r in relation_all:
    #     sig[r] = set()
    # for triple in triple_list:
    #     h = triple[0]
    #     r = triple[1]
    #     t = triple[2]
    #     if ('has_head', type_entity.get(h)) not in sig.get(r):
    #         sig.get(r).add(('has_head', type_entity.get(h)))
    #     if ('has_tail', type_entity.get(t)) not in sig.get(r):
    #         sig.get(r).add(('has_tail', type_entity.get(t)))
    # for r in relation_all:
    #     if sig.get(r) not in L:
    #         L.append(sig.get(r))
    return type_entity  # dict{e:type}


def corrupt(data_pos, data_all, flag=1):
    # train_data_pos=[]
    train_data_neg = []
    # df_train = pd.DataFrame(data_pos,columns=['head', 'relation', 'tail', 'timestamp'])
    df_all = pd.DataFrame(data_all, columns=['h', 'r', 't', 'timestamp'])
    data_all_set = {tuple((item[0], item[1], item[2], item[3])) for item in data_all}
    entity_num = max(set(df_all['h'].tolist()) | set(df_all['t'].tolist())) + 1
    for item in tqdm(data_pos):
        head = item[0]
        relation = item[1]
        tail = item[2]
        timestamp = item[3]
        count = 0
        if (flag == 1):
            while count < 3:
                neg_tail = np.random.randint(0, entity_num)
                if df_all[(df_all['h'] == head) & (df_all['r'] == relation) & (df_all['t'] == neg_tail) & (
                        df_all['timestamp'] == timestamp)].shape[0] == 0:
                    train_data_neg.append((head, relation, neg_tail, timestamp))
                    count += 1

        elif (flag == 2):
            # df_tmp = data_df[['H', 'R', 'T']]
            # tmp_dict = df_tmp.groupby(by=['H', 'R']).groups  # candiate entities for negative sampling
            # candidate_dict = {}
            # for key in tmp_dict:
            #     idx_list = list(tmp_dict.get(key))
            #     tail_list = list(set(df_tmp.iloc[idx_list]['T'].tolist()))
            #     candidate_dict[key] = tail_list
            pass
        elif (flag == 3):
            while count < 3:
                neg_tail = np.random.randint(0, entity_num)
                if df_all[
                    (df_all['h'] == head) & (df_all['r'] == relation) & (df_all['t'] == neg_tail)].shape[
                    0] == 0:
                    train_data_neg.append((head, relation, neg_tail, timestamp))
                    count += 1

    train_data_pos = data_pos
    train_data_neg = np.array(train_data_neg)
    return train_data_pos, train_data_neg

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
            quadruple_list.append((head, relation, tail,timestamp))
            timestamp_all.add(timestamp)
    timestamp_all = list(timestamp_all)
    timestamp_all.sort()
    return np.asarray(quadruple_list), np.asarray(timestamp_all)

def process_dataset(dataset):
    print("###############load dataset: {}################".format(dataset))

    path = os.path.join(root_path, 'data/{}'.format(dataset))
    if dataset == 'ICEWS14':
        train_data, train_times = load_quadruples(path, 'train.txt')
        test_data, test_times = load_quadruples(path, 'test.txt')
        valid_data = None
        data = np.concatenate([train_data, test_data])
    else:
        train_data, train_times = load_quadruples(path, 'train.txt')
        test_data, test_times = load_quadruples(path, 'test.txt')
        valid_data, valid_times = load_quadruples(path, 'valid.txt')
        data = np.concatenate([train_data, valid_data, test_data])

    df = pd.DataFrame(data, columns=['h', 'r', 't', 'timestamp'])
    df.to_csv(
        os.path.join(root_path, 'data_cached_new/{}/df_all_data.csv'.format(dataset)), index=False)

    type_entity = type_acquisition(dataset)  # entity: type
    # print(max(type_entity.values()))
    # df_e_type = pd.DataFrame(columns=['entity', 'type'])
    # df_e_type['entity'] = list(type_entity)
    # df_e_type['type'] = list(type_entity.values())

    for flag in [1, 3]:
        train_data_pos, train_data_neg = corrupt(train_data, data, flag=flag)
        df_train_data_neg = pd.DataFrame(train_data_neg, columns=['h', 'r', 't', 'timestamp'])
        df_train_data_neg.to_csv(
            os.path.join(root_path, 'data_cached_new/{}/df_train_data_neg_flag_{}.csv'.format(dataset, flag)),
            header=0)

    df_train_data_pos = pd.DataFrame(train_data, columns=['h', 'r', 't', 'timestamp'])
    h_type = [type_entity.get(x) for x in df_train_data_pos['h'].tolist()]
    t_type = [type_entity.get(x) for x in df_train_data_pos['t'].tolist()]
    df_train_data_pos['h_type'] = h_type
    df_train_data_pos['t_type'] = t_type

    df_valid_data = pd.DataFrame(valid_data, columns=['h', 'r', 't', 'timestamp'])
    h_type = [type_entity.get(x) for x in df_valid_data['h'].tolist()]
    t_type = [type_entity.get(x) for x in df_valid_data['t'].tolist()]
    df_valid_data['h_type'] = h_type
    df_valid_data['t_type'] = t_type

    df_test_data = pd.DataFrame(test_data, columns=['h', 'r', 't', 'timestamp'])
    h_type = [type_entity.get(x) for x in df_test_data['h'].tolist()]
    t_type = [type_entity.get(x) for x in df_test_data['t'].tolist()]
    df_test_data['h_type'] = h_type
    df_test_data['t_type'] = t_type

    df_train_data_pos.to_csv(os.path.join(root_path, 'data_cached_new/{}/df_train_data_pos.csv'.format(dataset)), index=False)
    df_valid_data.to_csv(os.path.join(root_path, 'data_cached_new/{}/df_valid_data.csv'.format(dataset)), index=False)
    df_test_data.to_csv(os.path.join(root_path, 'data_cached_new/{}/df_test_data.csv'.format(dataset)), index=False)


def dataloader(dataset, flag =1):
    file_path = os.path.join(root_path, 'data_cached_new/{}/df_train_data_pos.csv'.format(dataset))
    # print(file_path)
    # flag = 1  #
    if os.path.isfile(file_path):
        print('loading {} directly...'.format(dataset))
        train_data_pos = pd.read_csv(
            os.path.join(root_path, 'data_cached_new/{}/df_train_data_pos.csv'.format(dataset)),
            header=0)
        train_data_neg = pd.read_csv(
            os.path.join(root_path, 'data_cached_new/{}/df_train_data_neg_flag_{}.csv'.format(dataset, flag)),
            header=0)
        valid_data = pd.read_csv(os.path.join(root_path, 'data_cached_new/{}/df_valid_data.csv'.format(dataset)),
                                 header=0)
        test_data = pd.read_csv(os.path.join(root_path, 'data_cached_new/{}/df_test_data.csv'.format(dataset)),
                                header=0)
        all_data = pd.read_csv(os.path.join(root_path, 'data_cached_new/{}/df_all_data.csv'.format(dataset)),
                                header=0)
    else:
        print('>>>data not exits, processing ...')
        process_dataset(dataset)
        train_data_pos = pd.read_csv(
            os.path.join(root_path, 'data_cached_new/{}/df_train_data_pos.csv'.format(dataset)),
            header=0)
        train_data_neg = pd.read_csv(
            os.path.join(root_path, 'data_cached_new/{}/df_train_data_neg_flag_{}.csv'.format(dataset, flag)),
            header=0)
        valid_data = pd.read_csv(os.path.join(root_path, 'data_cached_new/{}/df_valid_data.csv'.format(dataset)),
                                 header=0)
        test_data = pd.read_csv(os.path.join(root_path, 'data_cached_new/{}/df_test_data.csv'.format(dataset)),
                                header=0)
        all_data = pd.read_csv(os.path.join(root_path, 'data_cached_new/{}/df_all_data.csv'.format(dataset)),
                               header=0)
    num_E, num_R, num_type = get_statistics(dataset)
    return train_data_pos, train_data_neg, valid_data, test_data,all_data,num_E, num_R, num_type






