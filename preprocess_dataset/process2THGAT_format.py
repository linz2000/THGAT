import numpy as np
import pandas as pd
import sys


def preprocess(data_name, feat_dim):
    u_list, i_list, ts_list, label_list = [], [], [], []
    # feat_l = []
    idx_list = []
    edge_feat_l = []
    node_feat_dict = dict()

    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            ts = float(e[2])
            label = int(e[3])

            # feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            # feat_l.append(feat)

            if u not in node_feat_dict:
                node_feat_dict[u] = np.random.randn(feat_dim)
            if i not in node_feat_dict:
                node_feat_dict[i] = np.random.randn(feat_dim)
            edge_feat_l.append(node_feat_dict[u]*node_feat_dict[i])

    node_feat_dict_l = [(n, f) for n, f in node_feat_dict.items()]
    sorted(node_feat_dict_l,key=lambda x:x[0])
    # print(node_feat_dict_l)
    node_feat_l = [f for _,f in node_feat_dict_l]


    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(node_feat_l), np.array(edge_feat_l)


def processs_data(dataset, feat_dim):
    # randomly generate node features

    PATH = "../processed/{}/{}.csv".format(dataset, dataset)
    OUT_DF = '../processed/{}/ml.csv'.format(dataset)
    OUT_EDGE_FEAT = '../processed/{}/ml.npy'.format(dataset)
    OUT_NODE_FEAT = '../processed/{}/ml_node.npy'.format(dataset)

    df, node_feat, edge_feat = preprocess(PATH, feat_dim)
    df.idx += 1

    empty = np.zeros(feat_dim)[np.newaxis, :]
    node_feat = np.vstack([empty, node_feat])
    edge_feat = np.vstack([empty, edge_feat])
    print("node feat shape:", node_feat.shape)
    print("edge feat shape:", edge_feat.shape)

    df.to_csv(OUT_DF)
    np.save(OUT_NODE_FEAT, node_feat)
    np.save(OUT_EDGE_FEAT, edge_feat)


# python process2THGAT_format.py dataset 128

dataset = sys.argv[1]
feat_dim = int(sys.argv[2])
processs_data(dataset, feat_dim)

