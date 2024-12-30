from gen_modules import *
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class Gen():
    def __init__(self, usernum, itemnum, args, reuse=tf.AUTO_REUSE):  # reuse指是否重用模型参数

        # --------------设置占位符,用于后续填入直接运行模型程序---------------------
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg

        # ---mask变量是将input_seq中非零元素位置标记为1,并增加一个维度来掩盖不需要计算的部分---
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        # -----------------------------设置变量作用域--------------------------------
        with tf.variable_scope("SASRec_gen", reuse=reuse):
            # seq是input_seq经过嵌入得到的嵌入矩阵, item_emb_table是所有项目的嵌入矩阵
            # seq shape(None,maxlen,args.item_hidden_units)
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.item_hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings_gen",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            self.item_emb_table = item_emb_table

            # -------------------t是针对序列的位置信息进行一个嵌入,pos_emb_table是每个位置的嵌入向量------------------------
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.item_hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos_gen",
                reuse=reuse,
                with_t=True
            )
            # ----------------------------------------------------------------------------------------------------

            u0_latent = embedding(self.u[0],
                                  vocab_size=usernum + 1,
                                  num_units=args.user_hidden_units,
                                  zero_pad=False,
                                  scale=True,
                                  l2_reg=args.l2_emb,
                                  scope="user_embeddings_gen",
                                  with_t=False,
                                  reuse=reuse
                                  )
            # ----------------------------------------------------------------------

            # 对用户user进行embedding嵌入 u_latent是对批次用户进行嵌入得到的结果(batchsize,args.user_hidden_units)
            u_latent, user_emb_table = embedding(self.u,
                                                 vocab_size=usernum + 1,
                                                 num_units=args.user_hidden_units,
                                                 zero_pad=False,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="user_embeddings_gen",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            self.user_emb_table = user_emb_table

            self.hidden_units = args.item_hidden_units

            self.seq += t

            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.gen_dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            for i in range(args.gen_num_blocks):
                with tf.variable_scope("num_blocks_gen_%d" % i):
                    # self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                    #                       dropout_rate=args.gen_dropout_rate, is_training=self.is_training)

                    self.seq, attention = multihead_attention(queries=normalize(self.seq),
                                                              keys=self.seq,
                                                              num_units=self.hidden_units,
                                                              num_heads=args.gen_num_heads,
                                                              dropout_rate=args.gen_dropout_rate,
                                                              is_training=self.is_training,
                                                              causality=True,
                                                              scope="self_attention_gen")

                    # self.attention.append(attention)

                    # Feed forward
                    self.seq = feedforward2(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                                            dropout_rate=args.gen_dropout_rate, is_training=self.is_training)
                    self.seq *= mask
            # -----------------------------------------------------------------------------

            # num_blocks = 2   ## 2
            # block_size = args.maxlen // num_blocks
            # weighted_blocks = []
            # # 每个块的自注意力次数
            # attention_repeats = [5,5]  # 假设每个块的自注意力次数不同  //44  (0.7304130356739287, 0.8711764705882353, 0.6847862978524737,  attention_repeats = [4,4]
            #
            # # 生成随机的块权重，范围在 0 到 1 之间
            # random_block_weights = tf.random.uniform([num_blocks], minval=0, maxval=1, dtype=tf.float32)
            #
            # # 归一化权重，使它们的和为 1
            # random_block_weights /= tf.reduce_sum(random_block_weights)
            #
            # decay_rate = 0.9
            # # ----------------------对具有用户嵌入的seq进行自注意力计算--------------------------
            # self.attention = []
            # # 预先计算权重
            # # block_weights = []
            # # for i in range(num_blocks):
            # #     block_weight = tf.exp(-decay_rate * (num_blocks - i - 1))
            # #     block_weights.append(block_weight)
            # # block_weights = tf.convert_to_tensor(block_weights, dtype=tf.float32)
            #
            #
            # for i in range(num_blocks):
            #     with tf.variable_scope("num_blocks_gen_%d" % i):
            #         start_idx = i * block_size
            #         end_idx = (i+1) * block_size
            #
            #         block_seq = self.seq[:,start_idx:end_idx,:]
            #         block_mask = mask[:, start_idx:end_idx, :]
            #
            #         # 多次自注意力操作
            #         for _ in range(attention_repeats[i]):
            #             block_seq, attention = multihead_attention(queries=normalize(block_seq),
            #                                                                             keys=block_seq,
            #                                                                             num_units=self.hidden_units,
            #                                                                             num_heads=args.gen_num_heads,
            #                                                                             dropout_rate=args.gen_dropout_rate,
            #                                                                             is_training=self.is_training,
            #                                                                             causality=True,
            #                                                                             scope="self_attention_gen_block_%d" % i)
            #             self.attention.append(attention)
            #
            #             # Apply feed forward layer within the block
            #             block_seq = feedforward2(normalize(block_seq), num_units=[self.hidden_units, self.hidden_units],
            #                                      dropout_rate=args.gen_dropout_rate, is_training=self.is_training)
            #             block_seq *= block_mask
            #
            #         # Apply weight to the block
            #         # block_weights = self.exponential_decay_weight(tf.cast(i, tf.float32), num_blocks)
            #         # weighted_block_seq = block_seq * block_weights[i]
            #         weighted_block_seq = block_seq * random_block_weights[i]
            #
            #         weighted_blocks.append(weighted_block_seq)
            #
            # # 拼接加权后的分块序列
            # self.seq_block_concat = tf.concat(weighted_blocks, axis=1)
            #
            # #整合信息：卷积
            # self.seq_conv = tf.layers.conv1d(inputs=self.seq_block_concat, filters=self.hidden_units, kernel_size=1,
            #                                padding='valid', activation=tf.nn.relu)
            #
            # # 可学习的残差连接权重
            # # alpha = tf.get_variable("alpha", shape=[], initializer=tf.constant_initializer(0.9))
            #
            # self.seq = 0.9 * self.seq + 0.1 * self.seq_conv
            #
            # self.seq = tf.layers.dropout(self.seq, rate=args.gen_dropout_rate, training=self.is_training)
            #
            # self.seq = feedforward2(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
            #                              dropout_rate=args.gen_dropout_rate, is_training=self.is_training)
            #
            # self.seq *= mask

        self.rewards = tf.placeholder(tf.float32, shape=(args.gen_batch_size * args.maxlen))

        # -------对 pos neg进行reshape,(None * args.maxlen) 方便后续计算--------
        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])

        # 从 item_emb_table查表最后获得的pos嵌入和neg嵌入分别是(None * args.maxlen , args.item_hidden_units)
        pos_emb = tf.nn.embedding_lookup(self.item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(self.item_emb_table, neg)

        # 将seq进行reshape操作-> (None * args.maxlen , hidden_units)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, self.hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        # 将得到的测试item通过查表item_emb_table来获取test_item的嵌入向量 (101 , args.item_hidden_units)
        test_item_emb = tf.nn.embedding_lookup(self.item_emb_table, self.test_item)

        # 将测试项目的嵌入向量与原始输入后经过一系列处理的seq_emb进行相乘(矩阵相乘),为了得出预测的项目与原始序列之间的相关性(打分)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        # self.test_logits所代表的是(我先忽略batchsize,每个位置的物品id对这101个项目分别做一个打分)
        # 计算结果取最后一个时间步的预测结果 [batch_size,101]
        self.test_logits = self.test_logits[:, -1, :]

        self.seq_emb_i = self.seq[:, -1, :]

        self.last_item_logits = tf.matmul(self.seq_emb_i, tf.transpose(self.item_emb_table))

        # prediction layer 计算相关性(通过内积的形式)  [batch_size * maxlen]  每个位置的样本对正例/负例的相关性得分
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # --------------------------------------ignore padding items (0)---[batchsize * maxlen]--------------------------------------
        istarget_pos = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        istarget_neg = tf.reshape(tf.to_float(tf.not_equal(neg, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])

        # -----------------------------------------设置预训练损失-------------------------------------------
        self.pre_loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget_pos -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget_neg
        ) / (tf.reduce_sum(istarget_pos) + tf.reduce_sum(istarget_neg))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.pre_loss += sum(reg_losses)

        self.pre_global_step = tf.Variable(0, name='global_step_gen', trainable=False)
        self.pre_optimizer = tf.train.AdamOptimizer(learning_rate=args.gen_lr, beta2=0.98)
        self.pre_train_op = self.pre_optimizer.minimize(self.pre_loss, global_step=self.pre_global_step)
        # ----------------------------------------------------------------------------------------------

        # ----------------------------设置生成器训练损失(额外增加了奖励机制)----------------------------------
        self.gen_loss = tf.reduce_sum(
            (- tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget_pos * self.rewards -
             tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget_neg)
        ) / (tf.reduce_sum(istarget_pos) + tf.reduce_sum(istarget_neg))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.gen_loss += sum(reg_losses)

        self.gen_global_step = tf.Variable(0, name='global_step_gen', trainable=False)
        self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=args.gen_lr, beta2=0.98)
        self.gen_train_op = self.gen_optimizer.minimize(self.gen_loss, global_step=self.gen_global_step)
        # ----------------------------------------------------------------------------------------------
        self.merged = tf.summary.merge_all()

    def exponential_decay_weight(block_idx, num_blocks, decay_rate=0.9):
        block_idx = tf.cast(block_idx, tf.float32)
        num_blocks = tf.cast(num_blocks, tf.float32)
        return tf.exp(-decay_rate * (num_blocks - block_idx - 1))

    def predict(self, sess, u, seq, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})

    def generate_position_k(self, sess, u, seq, k, args, batch=6040):
        print("sampling")
        sampled_item = np.zeros([len(u), args.maxlen, k], dtype=np.int32)
        for i in tqdm(range(batch), total=batch, ncols=70, leave=False, unit='u'):
            logit = sess.run(self.item_logits,
                             {self.u: u, self.input_seq: [seq[i]], self.is_training: False})
            logit = -logit
            index = logit.argsort()
            for position in range(args.maxlen):
                if seq[i][args.maxlen - 1 - position] == 0:
                    break
                cnt = 0
                for j in range(k):
                    if index[args.maxlen - 1 - position][cnt] == seq[i][args.maxlen - 1 - position]:
                        cnt += 1
                    sampled_item[i][args.maxlen - 1 - position][j] = index[args.maxlen - 1 - position][cnt]
                    cnt += 1

        return sampled_item  # user * maxlen * k

    def generate_k(self, sess, u, seq, pos, k):
        # sampled = gen_model.generate_k(sess, range(usernum), seq_total_train, seq_total[:, -1], k)
        # u代表完整用户序列,seq代表的是真实用户的训练+验证集,pos代表的是真实用户的测试集,k=10
        batch = 10
        # interval代表的是每个批次的序列长度
        interval = int(len(seq) / batch)
        begin = 0

        # sampled_item是想要为每个用户生成10个item
        sampled_item = np.zeros([len(u), k], dtype=np.int32)

        global_pos = 0

        for i in tqdm(range(batch - 1), total=batch - 1, ncols=70, leave=False, unit='u'):
            # --------------修改------------------------

            # logit 应该代表的是 这一批次中的验证集(单个)分别对所有item(itemnum)进行了一个打分
            logit = sess.run(self.last_item_logits,
                             {self.u: u[begin:begin + interval], self.input_seq: seq[begin:begin + interval],
                              self.is_training: False})
            begin += interval
            logit = -logit
            index = logit.argsort()
            for line in range(len(logit)):
                cnt = 0
                for rank in range(k):
                    if index[line][cnt] == pos[global_pos]:
                        cnt += 1
                    sampled_item[global_pos][rank] = index[line][cnt]
                    cnt += 1
                global_pos += 1

        logit = sess.run(self.last_item_logits,
                         {self.u: u[begin:len(u)], self.input_seq: seq[begin:len(seq)], self.is_training: False})
        for line in range(len(logit) - 1):
            cnt = 0
            for rank in range(k):
                if index[line][cnt] == pos[global_pos]: cnt += 1
                sampled_item[global_pos][rank] = index[line][cnt]
                cnt += 1
            global_pos += 1

        return sampled_item

    def generate_last_item(self, sess, u, seq):
        batch = 1
        interval = int(len(seq) / batch)
        begin = 0
        global_pos = 0
        top_item = np.zeros([len(u)])
        for i in range(batch - 1):
            logit = sess.run(self.last_item_logits,
                             {self.u: u, self.input_seq: seq[begin:begin + interval], self.is_training: False})
            begin += interval
            logit = -logit
            index = logit.argsort()
            for line in range(len(logit)):
                top_item[global_pos] = index[line][0]
                global_pos += 1
        logit = sess.run(self.last_item_logits,
                         {self.u: u, self.input_seq: seq[begin:len(seq)], self.is_training: False})
        logit = -logit
        index = logit.argsort()
        for line in range(len(logit)):
            top_item[global_pos] = index[line][0]
            global_pos += 1
        return top_item

    def generate_recommendations(self, sess, usernum, seq_total_train, seq_total, k, batch_size=128):
        recommendations = {}
        all_users = np.arange(1, usernum + 1)
        for i in tqdm(range(0, usernum, batch_size), total=usernum // batch_size, ncols=70, leave=False, unit='b'):
            batch_users = all_users[i:i + batch_size]
            batch_seq_train = seq_total_train[i:i + batch_size]
            batch_last_item = seq_total[i:i + batch_size, -1]

            # 获取logits
            logit = sess.run(self.last_item_logits,
                             {self.u: batch_users, self.input_seq: batch_seq_train, self.is_training: False})

            logit = -logit  # logits越大表示相关性越高，因此需要取负值
            index = logit.argsort()  # 对logits进行排序，返回排序后的索引

            batch_recommendations = np.zeros([len(batch_users), k], dtype=np.int32)
            for line in range(len(logit)):
                cnt = 0
                for rank in range(k):
                    while cnt < len(index[line]) and index[line][cnt] == batch_last_item[line]:
                        cnt += 1
                    if cnt < len(index[line]):
                        batch_recommendations[line][rank] = index[line][cnt]
                        cnt += 1

            for j, user in enumerate(batch_users):
                recommendations[user] = batch_recommendations[j]

        return recommendations
