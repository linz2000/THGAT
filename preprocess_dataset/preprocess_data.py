import sys
import os

# python preprocess_data.py dataset

dataset = sys.argv[1]

if not os.path.exists("../processed/{}".format(dataset)):
    os.makedirs("../processed/{}".format(dataset))

inputPath = "../raw_datasets/{}.txt".format(dataset)
outCsvPath = "../processed/{}/{}.csv".format(dataset, dataset)
outTxtPath = "../processed/{}/{}_new.txt".format(dataset, dataset)
outLabelPath = "../processed/{}/{}_origin_label.txt".format(dataset, dataset)
outNodeMapPath = "../processed/{}/{}_node_map.txt".format(dataset, dataset)

nlabelDict = dict()  # node label: 0, 1, 2,...
nlabel_idx = 0
elabelDict = dict()  # undirected edge, edge label: 0, 1, 2,...
elabel_idx = 0
nodeDict = dict()  # node id: 1, 2, 3,...
node_idx = 1
with open(inputPath) as idxF:
    for line in idxF:
        toks = line.strip().split('\t')
        l1 = toks[1]
        l2 = toks[3]
        if l1 not in nlabelDict:
            nlabelDict[l1] = str(nlabel_idx)
            nlabel_idx += 1
        if l2 not in nlabelDict:
            nlabelDict[l2] = str(nlabel_idx)
            nlabel_idx += 1

        str1 = nlabelDict[l1]+'_'+nlabelDict[l2]
        str2 = nlabelDict[l2]+'_'+nlabelDict[l1]
        if str1 not in elabelDict and str2 not in elabelDict:
            elabelDict[str1] = str(elabel_idx)
            elabelDict[str2] = str(elabel_idx)
            elabel_idx += 1

        n1 = toks[0]
        if n1 not in nodeDict:
            nodeDict[n1] = str(node_idx)
            node_idx += 1
        n2 = toks[2]
        if n2 not in nodeDict:
            nodeDict[n2] = str(node_idx)
            node_idx += 1
# print('elabelDict:', elabelDict)
# print('nlabelDict:', nlabelDict)
# print('len(nodeDict):', len(nodeDict))

# save node original label
with open(outLabelPath,'w') as outl:
    for l,idx in nlabelDict.items():
        outl.write(idx+"\t"+l+"\n")

# save node id map
with open(outNodeMapPath,'w') as outm:
    outm.write("nodeid + 1: "+str(len(nodeDict)+1)+"\n")
    for node,id in nodeDict.items():
        outm.write(node+"\t"+id+"\n")

inf = open(inputPath)
csvf = open(outCsvPath, 'w')
txtf = open(outTxtPath, 'w')

csvf.write("user_id,item_id,timestamp,state_label\n")
for line in inf:
    toks = line.strip().split('\t')
    n1 = nodeDict[toks[0]]
    n2 = nodeDict[toks[2]]
    l1 = nlabelDict[toks[1]]
    l2 = nlabelDict[toks[3]]
    ts = toks[4]
    str1 = l1+'_'+l2

    csvf.write(n1+','+n2+','+ts+','+elabelDict[str1]+'\n')
    txtf.write(n1+'\t'+l1+'\t'+n2+'\t'+l2+'\t'+ts+'\n')

csvf.close()
txtf.close()
inf.close()
