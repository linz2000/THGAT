import sys

def getMap(datapath, primepath, outpath):
    label_dict = dict()
    with open(datapath) as f:
        for line in f:
            toks = line.strip().split()
            if toks[1] in label_dict:
                label_dict[toks[1]] += 1
            else:
                label_dict[toks[1]] = 1

            if toks[3] in label_dict:
                label_dict[toks[3]] += 1
            else:
                label_dict[toks[3]] = 1
    label_list = [(l,m) for l,m in label_dict.items()]
    label_list = sorted(label_list,key=lambda x:x[1],reverse=True)
    # print(label_list)

    with open(primepath) as pf:
        line = next(pf)
        ps = line.strip().split(',')

    with open(outpath,"w") as outf:
        idx = 0
        for label,_ in label_list:
            outf.write("{}\t{}\n".format(label,ps[idx]))
            idx+=1

# python gen_prime_map.py dataset

dataset = sys.argv[1]
datapath = "../processed/{}/{}_new.txt".format(dataset, dataset)
primepath = "prime_map/prime1000.txt"
outpath = "prime_map/{}_map.txt".format(dataset)
getMap(datapath, primepath, outpath)
