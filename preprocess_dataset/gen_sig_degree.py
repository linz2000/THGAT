import networkx as nx
import numpy as np
import sklearn.preprocessing as pp
import argparse


def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Generate type degree based node signature.")

    parser.add_argument('--dataset', nargs='?', default='cora',
                        help='Input graph dataset')

    parser.add_argument('--mode_standard', type=int, default=0,
                        help='use pp.scale or PCA, default 0 is pp.scale, 1 is using PCA')

    return parser.parse_args()


class Sig:
    def __init__(self):
        self.old_g = nx.Graph()
        self.new_g = nx.Graph()
        self.sigs = dict()  # use self.gen_sig()
        self.type_map = dict()  # type to type id
        self.d = 0  # number of types

    def load_graph(self, graph_f):
        """
        load file into networkx
        :param graph_f: file path
        :return: nx.G
        """
        G = nx.Graph()
        f = open(graph_f)
        type_set = set()
        for line in f:
            toks = line.strip().split('\t')
            type_set.add(toks[1])
            type_set.add(toks[3])
            G.add_node(toks[0], label=toks[1])
            G.add_node(toks[2], label=toks[3])
            G.add_edge(toks[0], toks[2], label=toks[4])
        f.close()
        # set type2typeid
        self.d = type_set.__len__() + int(type_set.__len__() % 2)
        print("type_set: ", type_set.__len__())
        print("d: ", self.d)
        for i in range(type_set.__len__()):
            t = type_set.pop()
            self.type_map[t] = i

        return G

    def gen_sig(self):
        '''
        use self.new_g to generate signature list and save it in self.sigs
        self.sigs has been standardized
        '''
        G = self.new_g
        v_list = []
        sig_l = []  # all nodes' sig
        for v in G.nodes:
            v_list.append(v)
            tmp = np.zeros(self.d)
            vt = G.nodes[v]['label']
            tmp[self.type_map[vt]] += 1
            for u in G[v]:
                t = G.nodes[u]['label']
                tmp[self.type_map[t]] += 1
            sig_l.append(tmp)
        if args.mode_standard == 0:
            sig_l = pp.scale(sig_l)
        else:
            sig_l = pp.StandardScaler().fit_transform(sig_l)
        for i in range(len(v_list)):
            self.sigs[v_list[i]] = sig_l[i]

    def write_sig2file(self, out_f):
        '''
        write self.sigs to file, space as separator
        '''
        with open(out_f, 'w') as f:
            f.write(str(len(self.sigs))+" "+str(len(self.sigs['1']))+"\n")
            for i in self.sigs:
                f.write(i+" "+" ".join(map(str, self.sigs[i]))+"\n")


def format_sig():
    with open(sig_path) as sig_f:
        firstLine = sig_f.readline().strip().split(' ')
        max_idx = int(firstLine[0])
        sig_feat = np.zeros((max_idx + 1, int(firstLine[1])))
        for line in sig_f:
            toks = list(map(float, line.strip().split(' ')))
            sig_feat[int(toks[0])] = toks[1:]

    np.save(format_sig_path, sig_feat)

# python gen_sig_degree.py --dataset cora
if __name__ == '__main__':

    # generate type degree based node signature
    args = parse_args()

    dataset = args.dataset
    input_path = '../processed/{}/{}_new.txt'.format(dataset, dataset)
    sig_path = '../processed/{}/{}_degree_sig.txt'.format(dataset, dataset)
    format_sig_path = '../processed/{}/ml_node_degree_sig.npy'.format(dataset)

    s = Sig()
    s.new_g = s.load_graph(input_path)
    s.gen_sig()
    s.write_sig2file(sig_path)
    format_sig()

