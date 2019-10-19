from multiprocessing import Process, Queue

import numpy as np


# original implimentation use
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


# custom random sampling method considering item access frequency
def random_neg(s, table):
    t = table.searchsorted(np.random.randint(table[-1])) + 1
    while t in s:
        t = table.searchsorted(np.random.randint(table[-1])) + 1
    return t


def sample_function(user_train,
                    user_num,
                    item_num,
                    cum_table,
                    batch_size,
                    max_len,
                    result_queue,
                    seed):
    def sample():

        user = np.random.randint(1, user_num + 1)
        while len(user_train[user]) <= 1:
            user = np.random.randint(1, user_num + 1)

        seq = np.zeros([max_len], dtype=np.int32)
        pos = np.zeros([max_len], dtype=np.int32)
        neg = np.zeros([max_len], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = max_len - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                # neg[idx] = random_neq(1, item_num + 1, ts)
                neg[idx] = random_neg(ts, cum_table)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    np.random.seed(seed)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self,
                 user,
                 user_num,
                 item_num,
                 cum_table,
                 batch_size=64,
                 max_len=10,
                 n_workers=1):

        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function,
                    args=(
                        user,
                        user_num,
                        item_num,
                        cum_table,
                        batch_size,
                        max_len,
                        self.result_queue,
                        np.random.randint(2e9)
                    )
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
