from dis_modules import *


class Dis():
    def __init__(self, usernum, itemnum, args, reuse=tf.AUTO_REUSE):

        # --------------设置占位符,用于后续填入直接运行模型程序---------------------
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.label = tf.placeholder(tf.float32, shape=(None, 2))
        # -------------------------------------------------------------------

        # -mask变量是将input_seq中非零元素位置标记为1,并增加一个维度来掩盖不需要计算的部分-
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        # -----------------------------设置变量作用域--------------------------------
        with tf.variable_scope("discriminator", reuse=reuse):

            # seq是input_seq经过嵌入得到的嵌入矩阵
            # seq shape(None,maxlen,args.item_hidden_units)
            self.seq = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.item_hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings_dis",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # -----------------------------------------------------------------------

            # -------------------t是针对序列的位置信息进行一个嵌入,pos_emb_table是每个位置的嵌入向量------------------------
            t = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.item_hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos_dis",
                reuse=reuse,
                with_t=True
            )
            # ------------------------------------------------------------------------------------------------------

            # # 对用户user进行embedding嵌入 u_latent是对批次用户进行嵌入得到的结果
            # u_latent = embedding(self.u,
            #                      vocab_size=usernum + 1,
            #                      num_units=args.user_hidden_units,
            #                      zero_pad=False,
            #                      scale=True,
            #                      l2_reg=args.l2_emb,
            #                      scope="user_embeddings_dis",
            #                      with_t=False,
            #                      reuse=reuse
            #                      )


            # ------------------------为u_latent增加一个维度(args.maxlen)-------------------------------

            self.hidden_units = args.item_hidden_units

            # -------------------------------将seq与u_latent进行拼接-----------------------------------


            # 最终的序列表示
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq, rate=args.dis_dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask

            # Build blocks
            for i in range(args.dis_num_blocks):
                with tf.variable_scope("num_blocks_dis_%d" % i):
                    # Self-attention
                    # self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units, self.hidden_units],
                    #                       dropout_rate=args.dis_dropout_rate, is_training=self.is_training)

                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.hidden_units,
                                                   num_heads=args.dis_num_heads,
                                                   dropout_rate=args.dis_dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=False,
                                                   scope="self_attention_dis")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_units,self.hidden_units],
                                           dropout_rate=args.dis_dropout_rate, is_training=self.is_training)
                    self.seq *= mask


            self.seq = normalize(self.seq)


            self.seq = self.seq[:, -1, :]  # 形状为 [ batch_size , hidden_units ]的张量


            l2_reg_lambda = 0.2
            l2_loss1 = tf.constant(0.0)
            with tf.name_scope("output1"):
                W1 = tf.Variable(tf.truncated_normal([self.hidden_units, 2], stddev=0.1), name="W1")
                b1 = tf.Variable(tf.constant(0.1, shape=[2]), name="b1")
                l2_loss1 += tf.nn.l2_loss(W1)
                l2_loss1 += tf.nn.l2_loss(b1)
                self.scores1 = tf.nn.xw_plus_b(self.seq, W1, b1, name="scores1")
                self.ypred_for_auc1 = tf.nn.softmax(self.scores1)


            self.ypred_for_auc = self.ypred_for_auc1

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores1, labels=self.label)

                self.loss = tf.reduce_mean(loss1) + \
                            l2_reg_lambda * (l2_loss1)

        if reuse is reuse:
            self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
            self.global_step = tf.Variable(0, name='global_step_dis', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.dis_lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()


