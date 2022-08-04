# THGAT
Temporal Heterogeneous Graph Representation Learning with Neighborhood Type Modeling.

- try a type degree based THGAT example:

   `python train.py --data_dir cora --prefix degree --uniform --patience 10 --mode_node_sig_feature 1 --gpu 0`
   
- try a type prime based THGAT example:

   `python train.py --data_dir cora --prefix prime --uniform --patience 10 --mode_node_sig_feature 0 --gpu 0`
