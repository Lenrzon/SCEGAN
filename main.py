import os
import time
import argparse
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from gen_mysampler import gen_myWarpSampler
from gen_sampler import gen_WarpSampler
from gen_model import Gen
from dis_model import Dis
from dis_sampler import dis_WarpSampler
from util import *
import networkx as nx


# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='default', required=True)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--pre_train_g', default=True, type=bool)
parser.add_argument('--hidden_units', default=50, type=int)  #原50
parser.add_argument('--gen_lr', default=0.001, type=float)
parser.add_argument('--dis_lr', default=0.001, type=float)
parser.add_argument('--gen_batch_size', default=128, type=int)
parser.add_argument('--dis_batch_size', default=16, type=int)
parser.add_argument('--gen_num_blocks', default=5, type=int)  # 原5
parser.add_argument('--dis_num_blocks', default=2, type=int)  #理论上略微提升 原2
parser.add_argument('--gen_num_heads', default=2, type=int)  # 原3   刚才1    NDCG：0.636   刚才2  NDCG：0.6376   刚才3 NDCG：0.6376
parser.add_argument('--dis_num_heads', default=2, type=int)   # 原始2
parser.add_argument('--gen_dropout_rate', default=0.2, type=float)
parser.add_argument('--dis_dropout_rate', default=0.25, type=float) #原0.25  刚才0.2
parser.add_argument('--num_gan_epochs', default=20, type=int) #this
parser.add_argument('--num_pre_generator', default=400, type=int)
parser.add_argument('--num_pre_discriminator', default=1, type=int)
parser.add_argument('--num_train_generator', default=20, type=int)
parser.add_argument('--num_train_discriminator', default=1, type=int)
parser.add_argument('--recommendations_num',default=200,type=int)

parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--dataset', default='MFGAN_final_beauty', required=True)

parser.add_argument('--k', default=10, type=int)
parser.add_argument('--gpu', default=0, type=int)

parser.add_argument('--user_hidden_units', default=50, type=int)
parser.add_argument('--item_hidden_units', default=150, type=int)
parser.add_argument('--threshold_user', default=0.8, type=float)
parser.add_argument('--threshold_item', default=0.8, type=float)

args = parser.parse_args()

k = 10
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)


def dis_expand_k(pos, sample_k):
    seq_temp = np.zeros(shape=(np.shape(pos)[0] * (k + 1), args.maxlen))
    label_temp = np.zeros(shape=(np.shape(pos)[0] * (k + 1), 2))
    for i in range(np.shape(pos)[0] * (k + 1)):
        if i % (k + 1) == 0:
            seq_temp[i] = pos[int(i / (k + 1))]
            label_temp[i] = [0, 1]
            for j in range(k):
                seq_temp[i + j + 1] = seq_temp[i]
                seq_temp[i + j + 1][-1] = sample_k[int(i / (k + 1))][j]
                label_temp[i + j + 1] = [1, 0]
    return seq_temp, label_temp


if __name__ == '__main__':

    # ------------------加载数据集 划分训练集、验证集、测试集-------------------------
    dataset = data_partition(args.dataset)
    [user_total, user_train, user_valid, user_test, usernum, itemnum] = dataset
    print("User number: %d,  Item number: %d." % (usernum, itemnum))
    # --------------------------------------------------------------------------

    # ----------------创建seq_total(用于存储所有用户的完整序列数据(训练+验证+测试)-------------------
    # -----------创建seq_total_train(存储除测试以外的序列数据(训练+验证)---------------------------
    seq_total = np.zeros([usernum, args.maxlen], dtype=np.int32)
    seq_total_train = np.zeros([usernum, args.maxlen], dtype=np.int32)
    for u in user_total:
        idx = args.maxlen - 2
        seq_total[u-1][idx+1] = user_total[u][-1]
        for i in reversed(user_total[u][:-1]):
            if idx == -1: break
            seq_total_train[u-1][idx+1] = i
            seq_total[u-1][idx] = i
            idx -= 1
        if idx == -1: seq_total_train[u-1][idx+1] = i
    # ---------------------------------------------------------------------------


    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    # ---------------------------------------------------------------------------

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f_alpha = open(os.path.join(args.dataset + '_' + args.train_dir, 'alpha.txt'), 'w')
    f_loss = open(os.path.join(args.dataset + '_' + args.train_dir, 'loss.txt') ,'w')
    f_loss_pre = open(os.path.join(args.dataset + '_' + args.train_dir, 'loss_pre.txt'), 'w')
    # ---------------------------------------------------------------------------------

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    # ---------------------------------------------------------------------------------------

    print('begin')
    gen_sampler = gen_WarpSampler(user_train,user_valid, usernum, itemnum, batch_size=args.gen_batch_size, maxlen=args.maxlen, threshold_user=args.threshold_user,
                threshold_item=args.threshold_item, n_workers=3)

    gen_model = Gen(usernum, itemnum, args)

    num_gen_batch = len(user_train) / args.gen_batch_size
    num_gen_batch = round(num_gen_batch)

    dis_sampler = dis_WarpSampler(user_total, usernum, itemnum,batch_size=args.dis_batch_size, maxlen=args.maxlen, n_workers=3)
    dis_model = Dis(usernum, itemnum, args)
    # ---------------------------------------

    sess.run(tf.global_variables_initializer())

    print('begin')

    # Pre-train generator
    variables = tf.contrib.framework.get_variables_to_restore()
    variables_to_restore = [v for v in variables if v.name.split('/')[0] == 'SASRec_gen']
    gen_saver = tf.train.Saver(variables_to_restore)

    if args.pre_train_g:
        for epoch in range(1, args.num_pre_generator + 1):
            print('Pre-training generator epoch: %d' % epoch)
            for step in tqdm(range(num_gen_batch), total=num_gen_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg, total = gen_sampler.next_batch()

                # 可以在这个地方改吗？
                loss, _ = sess.run([gen_model.pre_loss, gen_model.pre_train_op],
                                  {gen_model.u: u, gen_model.input_seq: seq, gen_model.pos: pos, gen_model.neg: neg,
                                   gen_model.is_training: True})
            f_loss_pre.write("epoch " + str(epoch) + ': ' + str(loss) + '\n')
            f.flush()
            if epoch % 100 == 0:
                print('Evaluating pre-train process', )
                t_test = evaluate(gen_model, dataset, args, sess)
                print('epoch:%d, NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f. Sparse: NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f'
                      % (epoch, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))
                f.write(str(t_test) + '\n')
                f.flush()
        gen_saver.save(sess, "models_" + args.dataset + "/generator")
    else:
        gen_saver.restore(sess, "./models_" + args.dataset + "/generator")
        print('Evaluating pre-train process', )
        t_test = evaluate(gen_model, dataset, args, sess)
        print('NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f. Sparse: NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f'
              % (t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))
        f.write(str(t_test) + '\n')
        f.flush()

    # Pre-train discriminator
    num_dis_batch = len(user_total) / args.dis_batch_size
    num_dis_batch = round(num_dis_batch)

    print('Sampling......')
    sampled = gen_model.generate_k(sess, range(usernum), seq_total_train, seq_total[:, -1], k)
    for epoch in range(1, args.num_pre_discriminator + 1):
        print('Pre-training discriminator epoch: %d' % epoch)
        tot_loss = 0
        for step in tqdm(range(num_dis_batch), total=num_dis_batch, ncols=70, leave=False, unit='b'):
            u, pos= dis_sampler.next_batch()
            seq_train, label= dis_expand_k(pos, [sampled[user-1] for user in u])
            loss, _ = sess.run([dis_model.loss, dis_model.train_op],
                               {dis_model.u: u, dis_model.input_seq: seq_train, dis_model.label: label,
                                dis_model.is_training: True})
            tot_loss += loss

    loss_tot = []
    cnt_loss = -1
    maxNDCG = -1
    avgNDCG = 0
    avgHR = 0
    avgMRR = 0

    recommendations = gen_model.generate_recommendations(sess, usernum, seq_total_train, seq_total,
                                                          args.recommendations_num)  ##seq_total or seq_total_train
    gen_mysampler = gen_myWarpSampler(user_train, usernum, itemnum, recommendations, batch_size=args.gen_batch_size,
                                     maxlen=args.maxlen,n_workers=3)

    # Adversarial training
    for turn in range(1, args.num_gan_epochs + 1):

        # Train the generator
        for epoch in range(1, args.num_train_generator + 1):  # num_train_generator:80
            print('Training turn %d generator epoch: %d' % (turn, epoch))
            cnt_loss += 1
            loss_tot.append(0)
            for step in tqdm(range(num_gen_batch), total=num_gen_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg, total = gen_sampler.next_batch()
                predicts = gen_model.generate_last_item(sess, u, seq)
                total = np.array(total)
                total[:, -1] = predicts
                rewards = sess.run(dis_model.ypred_for_auc,
                                   {dis_model.u: u, dis_model.input_seq: total, dis_model.is_training: False})
                rewards = np.array([item[1] for item in rewards])  # batch_size * 1
                rewards = np.reshape(np.repeat(np.expand_dims(rewards, axis=1), args.maxlen, axis=1),
                                     [len(seq) * args.maxlen])
                loss, _ = sess.run([gen_model.gen_loss, gen_model.gen_train_op],
                                   {gen_model.u: u, gen_model.input_seq: seq, gen_model.pos: pos, gen_model.neg: neg,
                                    gen_model.rewards: rewards, gen_model.is_training: True})
                f_loss.write('step: ' + str(step) + ':' + str(loss) + '\n')
                f_loss.flush()
                loss_tot[cnt_loss] += loss
            loss_tot[cnt_loss] /= num_gen_batch

            if epoch % 10 == 0:
                print('Evaluating adversarial process', )
                t_test = evaluate(gen_model, dataset, args, sess)
                avgNDCG+=t_test[0]
                avgHR+=t_test[1]
                avgMRR+=t_test[2]
                if(maxNDCG < t_test[0]):
                    maxNDCG = t_test[0]
                print('epoch:%d, NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f. Sparse: NDCG@10: %.4f, HR@10: %.4f, MRR: %.4f'
                      % (epoch, t_test[0], t_test[1], t_test[2], t_test[3], t_test[4], t_test[5]))
                f.write('turn: ' + str(turn) + ',  ' + str(t_test) + '\n')
                f.flush()

        # Train the discriminator
        print('Sampling......')
        sampled = gen_model.generate_k(sess, range(usernum), seq_total_train, seq_total[:, -1], k)
        for epoch in range(1, args.num_train_discriminator + 1):  # num_train_discriminator:1
            print('Training discriminator epoch: %d' % epoch)
            tot_loss = 0
            for step in tqdm(range(num_dis_batch), total=num_dis_batch, ncols=70, leave=False, unit='b'):
                u, pos = dis_sampler.next_batch()
                # 看一眼形状
                seq_train, label= dis_expand_k(pos, [sampled[user - 1] for user in u])
                loss, _ = sess.run([dis_model.loss, dis_model.train_op],
                                   {dis_model.u: u, dis_model.input_seq: seq_train, dis_model.label: label,
                                    dis_model.is_training: True})
                tot_loss += loss

    np.save('loss.npy', loss_tot)
    f.close()
    f_alpha.close()
    f_loss.close()
    print("Done")
    print(maxNDCG)
    print(avgNDCG/40,avgHR/40,avgMRR/40)

    gen_sampler.close()
    dis_sampler.close()