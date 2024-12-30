import random
import numpy as np
from multiprocessing import Process, Queue
# -*- coding: utf-8 -*-#

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t



def sample_function(user_train,user_vaild, usernum, itemnum, batch_size, maxlen, threshold_user, threshold_item, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)  # 随机user
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)  # 如果user训练集长度小于等于1,重新随机user
        # while len(user_train[user]) >= 150:
        #     user = np.random.randint(1, usernum + 1)  # 如果user训练集长度小于等于1,重新随机user


        # 定义seq(序列)、pos(正例)、neg(负例)、total(完整序列)
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        total = np.zeros([maxlen], dtype=np.int32)



        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])  # 设置set集合用于方便选取负例

        # pre_nxt = nxt
        # flag = 0
        # total保存的是user的完整训练集,seq保存的是除去最后一位训练集的序列,pos保存的是除去最初一位训练集的序列
        # 如果random.random() > threshold_item:那么seq与pos在该位将随机生成一个项目(用于SSE)
        # neg选取user训练集中不存在的项目
        #
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i # 除去最后一位
            # total[idx] = pre_nxt
            if random.random() > threshold_item:
                # pre_nxt = i
                i = np.random.randint(1,itemnum)
                nxt = np.random.randint(1,itemnum)
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
     #       if flag == 0 :
     #           pre_nxt = nxt
     #       else :
     #           flag = 0
            idx -= 1
            if idx == -1: break
        if idx != -1:
            total[idx] = nxt

        # if random.random() > 0.8:
        #     user = np.random.randint(1,usernum+1)

        return (user ,seq, pos, neg, total)




    np.random.seed(SEED)  # 设置随机数种子
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

# gen_WarpSampler用于并行生成训练样本
class gen_WarpSampler(object):
    def __init__(self, User,User1, usernum, itemnum, batch_size=64, maxlen=10,
                 threshold_user=1.0, threshold_item=1.0, n_workers=1):

        # -------------------------设置存储队列、进程列表----------------------------
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        # -----------------------------------------------------------------------


        # --------------------对于每一个worder,都执行采样函数------------------------
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,User1,
                                                      usernum,
                                                      itemnum,

                                                      batch_size,
                                                      maxlen,
                                                      threshold_user,
                                                      threshold_item,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True  # 设置进程为守护进程,主进程结束,守护进程也就结束
            self.processors[-1].start()        # 启动该进程
        # -----------------------------------------------------------------------

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
