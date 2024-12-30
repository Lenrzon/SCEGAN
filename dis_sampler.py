import numpy as np
from multiprocessing import Process, Queue
import random

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum,  batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
        # while len(user_train[user]) >= 150:
        #     user = np.random.randint(1, usernum + 1)  # 如果user训练集长度小于等于1,重新随机user

        pos = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        ts = set(user_train[user])  # 设置set集合用于方便选取负例
        for i in reversed(user_train[user][:]):
            # 修改 应用SSE
            if random.random() > 0.8:
                pos[idx] = np.random.randint(1,itemnum)
            # -------------------------------------------#
            else :
                pos[idx] = i
            idx -= 1
            if idx == -1:
                break

        # if random.random() > 0.8:
        #    user = np.random.randint(1,usernum+1)
        return user, pos

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class dis_WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,

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
