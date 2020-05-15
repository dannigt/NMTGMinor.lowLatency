import sys
import math
from kaldiio import ReadHelper
from kaldiio import WriteHelper
import os
import pickle
import random
random.seed(0)
import os.path as path

if len(sys.argv) < 3:
    raise Exception('Not enough args!')

src_file = sys.argv[1]
out_dir = sys.argv[2]
transcript_file = sys.argv[3]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

num_partial_seqs = dict()

with open(transcript_file, 'r') as transcript:
    trans = transcript.readlines()

with ReadHelper('scp:' + src_file) as reader:
    with open(path.join(out_dir, 'text'), 'w') as new_trans:
        with WriteHelper('ark,scp:{0}/file.ark,{1}/feats.scp'.format(out_dir, out_dir)) as writer:
            cnt = 0
            # loop through all utterances
            for utt_id, feature_vectors in reader:
                proportion = round(random.random() * 0.4 + 0.1, 2)   # (0.1, 0.5)

                toks = trans[cnt].split(' ')

                toks = toks[:int(len(toks) * proportion)]

                num_frames = int(feature_vectors.shape[0] * proportion)

                print(proportion, toks)

                if len(toks) != 0:
                    # writing to ark, old
                    if random.random() < 0.4:
                        writer("_".join([utt_id, "0"]), feature_vectors)
                        new_trans.write(trans[cnt])
                    # writing to ark, new
                    writer("_".join([utt_id, "1"]), feature_vectors[:num_frames])
                    new_trans.write(' '.join(toks) + '\n')
                cnt += 1

print('Done.')
