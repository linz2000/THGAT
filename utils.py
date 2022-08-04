import numpy as np
import random

### Utility function and class
### adapted from  Da Xu et al./tgat
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, adj_list):
        self.adj_list = adj_list  #adj_list[i]--> adj nodes set
        self.node_num = len(adj_list)-1

    def sample(self, src_l):
        dst_l = list()
        for src_node in src_l:
            dst_node = random.randint(1,self.node_num)
            while dst_node in self.adj_list[src_node]:
                dst_node = random.randint(1, self.node_num)
            dst_l.append(dst_node)
        return src_l, np.array(dst_l)