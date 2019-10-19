import copy
import random
import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf


def data_partition(path: str):
    user_num = 0
    item_num = 0
    user = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    with open(path, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            user_num = max(u, user_num)
            item_num = max(i, item_num)
            user[u].append(i)

    item_count = np.zeros((item_num), dtype=np.uint64)
    for us in user:
        nfeedback = len(user[us])

        if nfeedback < 3:
            user_train[us] = user[us]
            user_valid[us] = []
            user_test[us] = []
        else:
            user_train[us] = user[us][:-2]
            user_valid[us] = []
            user_valid[us].append(user[us][-2])
            user_test[us] = []
            user_test[us].append(user[us][-1])

        for item in user[us]:
            item_count[item-1] += 1

    ns_exponent = 0.75
    cum_table = np.zeros_like(item_count)
    train_words_pow = np.sum(item_count**ns_exponent)
    cumulative = 0.0
    for idx in range(item_num):
        cumulative += item_count[idx]**ns_exponent
        cum_table[idx] = round(cumulative / train_words_pow * 2**63 - 1)

    return [user_train, user_valid, user_test, user_num, item_num, item_count, cum_table]


def evaluate(model, emb, dataset, max_len):
    [train, valid, test, usernum, itemnum, item_count, cum_table] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([max_len], dtype=np.int32)
        idx = max_len - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        test_emb = emb(np.pad(np.array(item_idx).reshape(-1, 1),
                       ((0, 0), (max_len-1, 0)), 'constant'))[:, -1]
        seq_emb = (-model.predict(seq.reshape(1, max_len)))[:, -1]
        predictions = tf.matmul(seq_emb, test_emb, transpose_b=True)
        predictions = predictions[0].numpy()

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('Validated {:.0f} users\r'.format(valid_user), end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, emb, dataset, max_len):
    [train, valid, test, usernum, itemnum, item_count, cum_table] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([max_len], dtype=np.int32)
        idx = max_len - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        valid_emb = emb(np.pad(np.array(item_idx).reshape(-1, 1),
                        ((0, 0), (max_len-1, 0)), 'constant'))[:, -1]
        seq_emb = (-model.predict(seq.reshape(1, max_len)))[:, -1]
        predictions = tf.matmul(seq_emb, valid_emb, transpose_b=True)
        predictions = predictions[0].numpy()

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('Validated {:.0f} users\r'.format(valid_user), end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
