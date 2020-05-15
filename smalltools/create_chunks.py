import sys
import math
from kaldiio import ReadHelper
from kaldiio import WriteHelper
import os
import pickle

if len(sys.argv) < 3:
    raise Exception('Not enough args!')

src_file = sys.argv[1]
out_dir = sys.argv[2]

if len(sys.argv) == 3:
    frames_per_segment = 100  # how many 10ms-frames per segment
else:
    frames_per_segment = int(sys.argv[3])


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

num_partial_seqs = dict()

with ReadHelper('scp:'+src_file) as reader:
    with WriteHelper('ark,scp:{0}/file.ark,{1}/feats.scp'.format(out_dir, out_dir)) as writer:
        # loop through all utterances
        for utt_id, feature_vectors in reader:
            # if '8kAWy2YodzQ_10' in utt_id:
                # utt_id, feature_vectors = next(audio_data)  # features
            n_segments = math.ceil(feature_vectors.shape[0] / frames_per_segment)

            for i in range(n_segments):
                seg_id = utt_id + "_" + str(i)
                # out feature
                new_feature_vecs = feature_vectors[:(i + 1) * frames_per_segment, :]

                # writing to ark
                writer(seg_id, new_feature_vecs)

            print('Done with {1} partial sequences of for {0}'.format(utt_id, n_segments))
            #break
            num_partial_seqs[utt_id] = n_segments
print('Done.')

with open(os.path.join(out_dir, "num.partial.seqs.{0}sec.pickle".format(0.01 * frames_per_segment)), 'wb') as f:
    pickle.dump(num_partial_seqs, f)
