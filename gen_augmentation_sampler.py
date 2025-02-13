import random

import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def random_neq_plus(l, r, s):
    t = np.random.randint(l, r)
    while t not in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, recommendations, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
        # while len(user_train[user]) >= 150: user = np.random.randint(1, usernum + 1)  # 如果user训练集长度小于等于1,重新随机user

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        total = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        com = set(recommendations[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i  # 除去最后一位
            total[idx] = nxt
            if random.random() > 0.8:

                if random.random() > 0.6:
                # pre_nxt = i
                # i = np.random.randint(1, itemnum + 1)
                # nxt = np.random.randint(1, itemnum + 1)
                    i = random_neq(1,itemnum+1,com)
                    nxt = random_neq(1,itemnum+1,com)
                else:
                    i = np.random.randint(1, itemnum)
                    nxt = np.random.randint(1, itemnum)

            seq[idx] = i
            pos[idx] = nxt  # 除去第一位
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break
        if idx != -1:
            total[idx] = nxt
        return user, seq, pos, neg, total

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


# gen_WarpSampler用于并行生成训练样本
class gen_myWarpSampler(object):
    def __init__(self, User, usernum, itemnum, recommendations, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      recommendations,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
