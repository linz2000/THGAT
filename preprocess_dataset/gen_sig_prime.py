from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import gc
import numpy as np

def get_map(filepath):
    type2prime = dict()
    with open(filepath) as mf:
        for line in mf:
            toks = line.strip().split()
            type2prime[toks[0]] = int(toks[1])
    return type2prime

def read_data(filepath, mapfile):
    type2prime = get_map(mapfile)
    with open(filepath) as f:
        for line in f:
            toks = line.strip().split("\t")
            node0id = toks[0]
            node0type = type2prime[toks[1]]
            nodeid = toks[2]
            nodetype = type2prime[toks[3]]

            if node0id not in neighbor_type_dict:
                neighbor_type_dict[node0id] = []
            if nodeid not in neighbor_type_dict:
                neighbor_type_dict[nodeid] = []
            neighbor_type_dict[node0id].append(nodetype)
            neighbor_type_dict[nodeid].append(node0type)

def get_sig(prime_list: list):
    d = 0
    vec = []

    sig = 1.
    for tp in prime_list:
        if d >= cutoff:
            return d, vec
        if sig * tp > sig_max:
            vec.append(sig)
            sig = tp
            d += 1
        else:
            sig *= tp
    if sig > 1 or len(vec) == 0:
        vec.append(sig)
        d += 1

    return d, vec

def get_sigs():
    d_max = 0
    vecs = dict()

    for node, prime_list in neighbor_type_dict.items():
        d, vec = get_sig(prime_list)
        vecs[node] = vec
        if d > d_max:
            d_max = d

    for node in vecs:
        for i in range(d_max - vecs[node].__len__()):
            vecs[node].append(0)

    return vecs, d_max

def decomp(n, v, n_max):
    print("target dimension:", n)
    print("max dimension:", n_max)
    vlist = []
    vidlist = []
    for vi in v:
        vlist.append(v[vi])
        vidlist.append(vi)

    ss = StandardScaler()
    if n > n_max:
        print("target dimension > max dimension")
        vlist = ss.fit_transform(vlist)
        return vlist, vidlist, False

    for x in list(locals().keys()):
        del locals()[x]
    gc.collect()

    pca = PCA(n_components=n)
    v_pca = pca.fit_transform(vlist)
    v_pca = ss.fit_transform(v_pca)
    return v_pca, vidlist, True

def format_sig(sigpath, outsigpath):
    with open(sigpath) as sig_f:
        firstLine = sig_f.readline().strip().split(' ')
        max_idx = int(firstLine[0])
        sig_feat = np.zeros((max_idx + 1, int(firstLine[1])))
        for line in sig_f:
            toks = list(map(float, line.strip().split(' ')))
            sig_feat[int(toks[0])] = toks[1:]

    np.save(outsigpath, sig_feat)


# python gen_sig_prime.py dataset 128 1000

if __name__ == "__main__":
    dataset = sys.argv[1]
    d_target = int(sys.argv[2])  # target signature dimension
    sig_max = int(sys.argv[3])  # max value of each dimension
    cutoff = 2000  # max length

    fpath = "../processed/{}/{}_new.txt".format(dataset, dataset)
    mpath = "prime_map/{}_map.txt".format(dataset)
    outpath = "../processed/{}/{}_prime_sig.txt".format(dataset, dataset)
    outsigpath = '../processed/{}/ml_node_prime_sig.npy'.format(dataset)

    neighbor_type_dict = dict()

    read_data(fpath, mpath)
    vec_sig, d_sig = get_sigs()
    vec_decomp, vid_list, is_pca = decomp(d_target, vec_sig, d_sig)

    extra_zero = d_target - len(vec_decomp[0])
    print("extra zero num:", extra_zero)

    with open(outpath, mode="w") as sigf:
        sigf.write(str(vec_decomp.__len__()))
        sigf.write(" ")
        sigf.write(str(d_target))
        sigf.write("\n")
        for i in range(vec_decomp.__len__()):
            sigf.write(str(vid_list[i]))
            for j in vec_decomp[i]:
                sigf.write(" ")
                sigf.write(str(j))
            for _ in range(extra_zero):
                sigf.write(" 0")
            sigf.write("\n")

    format_sig(outpath, outsigpath)
