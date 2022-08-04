import math
import logging
import time
import random
import sys
import argparse
import os

import torch
import pandas as pd
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import THGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler

### Argument and global variables
parser = argparse.ArgumentParser('Interface for THGAT experiments on link prediction')
parser.add_argument('--data_dir', type=str, help='data dir prefix to use, try cora, dblp_co, dblp_ac or tky', default='cora')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=128, help='Dimentions of the node embedding')
parser.add_argument('--time_dim', type=int, default=128, help='Dimentions of the time embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

parser.add_argument('--mode_node_feature', type=int, default=0, help='mode of node features, default 0 is random, 1 is using sig as nodeFeat')
parser.add_argument('--mode_node_sig_feature', type=int, default=1, help='mode of node sig features, 0 is using prime sig(d128), default 1 is using degree sig')
parser.add_argument('--patience', type=int, default=10, help='early stop to train the model')
parser.add_argument('--no_train', action='store_true', help='not training the model')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

random.seed(2020)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
data_dir = args.data_dir
mode_node_feature = args.mode_node_feature
mode_node_sig_feature = args.mode_node_sig_feature
PATIENCE = args.patience
IS_TRAIN = not args.no_train

if not os.path.exists("saved_models/{}".format(data_dir)):
    os.makedirs("saved_models/{}".format(data_dir))
if not os.path.exists("saved_checkpoints/{}".format(data_dir)):
    os.makedirs("saved_checkpoints/{}".format(data_dir))
MODEL_SAVE_PATH = f'./saved_models/{data_dir}/{args.prefix}-{args.agg_method}-{args.attn_mode}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{data_dir}/{args.prefix}-{args.agg_method}-{args.attn_mode}-{epoch}.pth'
if mode_node_feature == 1:    #使用签名作为节点特征
    if not os.path.exists("saved_models/{}_sig".format(data_dir)):
        os.makedirs("saved_models/{}_sig".format(data_dir))
    if not os.path.exists("saved_checkpoints/{}_sig".format(data_dir)):
        os.makedirs("saved_checkpoints/{}_sig".format(data_dir))
    MODEL_SAVE_PATH = f'./saved_models/{data_dir}_sig/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
    get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{data_dir}_sig/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

if not os.path.exists("saved_embs/{}".format(data_dir)):
    os.makedirs("saved_embs/{}".format(data_dir))
EMB_SAVE_PATH = f"saved_embs/{data_dir}/{args.prefix}_{data_dir}.emb"

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'log/{data_dir}_{args.prefix}_{time.time()}.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


def eval_one_epoch(hint, thgan, sampler, src, dst, ts, label=None):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        thgan = thgan.eval()
        TEST_BATCH_SIZE=30
        num_test_instance = len(src)
        while num_test_instance%TEST_BATCH_SIZE == 1:
            TEST_BATCH_SIZE += 1
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):

            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(src_l_cut)

            pos_prob, neg_prob = thgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)

### Load data and train val test split
g_df = pd.read_csv('./processed/{}/ml.csv'.format(data_dir))
e_feat = np.load('./processed/{}/ml.npy'.format(data_dir))
if mode_node_feature == 0:
    n_feat = np.load('./processed/{}/ml_node.npy'.format(data_dir))
else:
    n_feat = np.load('./processed/{}/ml_node_prime_sig.npy'.format(data_dir))
if mode_node_sig_feature == 1:
    s_feat = np.load('./processed/{}/ml_node_degree_sig.npy'.format(data_dir))
else:
    s_feat = np.load('./processed/{}/ml_node_prime_sig.npy'.format(data_dir))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_idx = max(src_l.max(), dst_l.max())

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
logger.info("val time: {}".format(val_time))
logger.info("test time: {}".format(test_time))

valid_train_flag = ts_l < val_time
valid_val_flag = (ts_l >= val_time) * (ts_l < test_time )
valid_test_flag = ts_l >= test_time

train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]

val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]

test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_e_idx_l = e_idx_l[valid_test_flag]


### Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data
train_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    train_adj_list[src].append((dst, eidx, ts))
    train_adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(train_adj_list, uniform=UNIFORM)

val_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    val_adj_list[src].append((dst, eidx, ts))
    val_adj_list[dst].append((src, eidx, ts))
val_ngh_finder = NeighborFinder(val_adj_list, uniform=UNIFORM)

test_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    test_adj_list[src].append((dst, eidx, ts))
    test_adj_list[dst].append((src, eidx, ts))
test_ngh_finder = NeighborFinder(test_adj_list, uniform=UNIFORM)

train_adj_set = [set() for _ in range(max_idx + 1)]
for src, dst in zip(train_src_l, train_dst_l):
    train_adj_set[src].add(dst)
    train_adj_set[dst].add(src)

adj_set = [set() for _ in range(max_idx + 1)]
for src, dst in zip(src_l, dst_l):
    adj_set[src].add(dst)
    adj_set[dst].add(src)

train_rand_sampler = RandEdgeSampler(train_adj_set)
val_rand_sampler = RandEdgeSampler(adj_set)
test_rand_sampler = RandEdgeSampler(adj_set)


start_time = time.time()
### Model initialize
device = torch.device('cuda:{}'.format(GPU))
thgan = THGAN(train_ngh_finder, n_feat, e_feat, sig_feat=s_feat,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
optimizer = torch.optim.Adam(thgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
thgan = thgan.to(device)

if IS_TRAIN:
    num_instance = len(train_src_l)
    TRAIN_BATCH_SIZE = BATCH_SIZE
    while num_instance%TRAIN_BATCH_SIZE == 1:
        TRAIN_BATCH_SIZE += 1
    num_batch = math.ceil(num_instance / TRAIN_BATCH_SIZE)
    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('training batch size: {}'.format(TRAIN_BATCH_SIZE))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    early_stopper = EarlyStopMonitor(max_round=PATIENCE)
    for epoch in range(NUM_EPOCH):
        # Training
        # training use only training graph
        thgan.ngh_finder = train_ngh_finder
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)
        logger.info('\n start {} epoch'.format(epoch))
        for k in range(num_batch):

            s_idx = k * TRAIN_BATCH_SIZE
            e_idx = min(num_instance, s_idx + TRAIN_BATCH_SIZE)
            src_l_cut, dst_l_cut = train_src_l[idx_list[s_idx:e_idx]], train_dst_l[idx_list[s_idx:e_idx]]
            ts_l_cut = train_ts_l[idx_list[s_idx:e_idx]]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(src_l_cut)

            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)

            optimizer.zero_grad()
            thgan = thgan.train()
            pos_prob, neg_prob = thgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)

            loss = criterion(pos_prob, pos_label)
            loss += criterion(neg_prob, neg_label)

            loss.backward()
            optimizer.step()
            # get training results
            with torch.no_grad():
                thgan = thgan.eval()
                pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                f1.append(f1_score(true_label, pred_label))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))

        # validation phase use all information
        thgan.ngh_finder = val_ngh_finder
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch('model val', thgan, val_rand_sampler, val_src_l,
        val_dst_l, val_ts_l)

        logger.info('epoch: {}:'.format(epoch))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train acc: {}, val acc:{}'.format(np.mean(acc), val_acc))
        logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
        logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))
        logger.info('train f1: {}, val f1: {}'.format(np.mean(f1), val_f1))

        if early_stopper.early_stop_check(val_auc):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            thgan.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            thgan.eval()
            break
        else:
            torch.save(thgan.state_dict(), get_checkpoint_path(epoch))

    # testing phase use all information
    thgan.ngh_finder = test_ngh_finder
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('model test', thgan, test_rand_sampler, test_src_l,
                                                          test_dst_l, test_ts_l)
    logger.info('Test statistics -- acc: {}, ap: {}, f1: {}, auc:{}'.format(test_acc,  test_ap, test_f1, test_auc))
    end_time = time.time()
    logger.info('Training time:{}'.format(end_time-start_time))

    logger.info('Saving THGAN model')
    torch.save(thgan.state_dict(), MODEL_SAVE_PATH)
    logger.info('THGAN model saved')
else:
    thgan.load_state_dict(torch.load(MODEL_SAVE_PATH))


#get node embeddings
logger.info('get node embeddings...')
max_ts = ts_l.max()
node_idx_l = np.arange(1, max_idx+1)
node_ts_l = np.array([max_ts]*max_idx)
thgan.eval()
with torch.no_grad():
    node_num = len(node_idx_l)
    BATCH_SIZE_TMP = BATCH_SIZE
    while(node_num%BATCH_SIZE_TMP == 1):
        BATCH_SIZE_TMP += 1
    emb_batch_num = math.ceil(node_num/BATCH_SIZE_TMP)
    emb_list = []

    for i in range(emb_batch_num):
        st = BATCH_SIZE_TMP*i
        ed = min(node_num,st+BATCH_SIZE_TMP)

        node_idx_cut = node_idx_l[st:ed]
        node_ts_cut = node_ts_l[st:ed]

        embs_cut = thgan.tem_conv(node_idx_cut, node_ts_cut, NUM_LAYER, NUM_NEIGHBORS)
        if torch.cuda.is_available():
            embs_cut = embs_cut.cpu().detach().numpy()
        else:
            embs_cut = embs_cut.detach().numpy()

        emb_list.append(embs_cut)
    embs = np.concatenate(emb_list)
    logger.info('done')

    #save embeddings to file
    logger.info('save node embeddings...')
    with open(EMB_SAVE_PATH,'w') as outf:
        for i in range(max_idx):
            outf.write(str(i+1)+" "+" ".join(str(x) for x in embs[i])+"\n")

    logger.info('done')

