import sys
import pickle

output_path = sys.argv[1]  #'/home/danni/dev.beam.t.base'  #
pkl_path = sys.argv[2]  #'/home/danni/workspace/data/how2/data/orig/how2-300h-v1/data/val_part/num.partial.seqs.0.5sec.pickle'  #

with open(pkl_path, 'rb') as f:
    num_partial_seqs = pickle.load(f)

with open(output_path, 'r') as f:
    all_out = f.readlines()

cnt = 0
all_num_toks = 0
all_lat = 0
for k in num_partial_seqs:
    print(k, num_partial_seqs[k])
    num_toks = len(all_out[cnt].split(' '))
    all_lat += num_toks * num_partial_seqs[k]
    all_num_toks += num_toks
    cnt += 1

print(all_lat / all_num_toks)