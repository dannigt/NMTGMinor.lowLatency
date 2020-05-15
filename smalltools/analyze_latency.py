import sys
import numpy as np
import warnings

latency_file = sys.argv[1]  #'/home/danni/workspace/src/NMTGMinor.private/translate_latency.out'

num_segments = 0
num_tokens = 0

num_segments_no_punct = 0
num_tokens_no_punct = 0

with open(latency_file, 'r') as f:
    for i, line in enumerate(f.readlines()):
        # vals = [0]
        # vals.extend([int(i) for i in line.strip().split(',')])
        vals = [int(i) for i in line.strip().split(',')]
        vals = np.array(vals)
        # print('# tokens', vals[-1])
        # print(vals)
        my_diff = np.diff(vals)
        if len(my_diff[my_diff < 0]) > 0:
            # foo = my_diff[my_diff < 0]
            print(vals)
            warnings.warn('decreasing!')
        #print(my_diff * np.arange(1, len(vals)))
        total_segments = np.sum(my_diff * np.arange(1, len(vals)))
        #print(total_segments / vals[-1])

        if i % 2 == 0:
            num_segments += total_segments
            num_tokens += vals[-1]
        else:
            num_segments_no_punct += total_segments
            num_tokens_no_punct += vals[-1]

print('Latency', num_segments / num_tokens)
print('Latency no punct', num_segments_no_punct / num_tokens_no_punct)
