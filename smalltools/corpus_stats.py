# number of words, chars, word freq., sentence freq. audio len

import sys
import re

pattern = re.compile('[\W_]+', re.UNICODE)

text_path = sys.argv[1]  #/home/danni/workspace/data/cv/clean_validated/text'
#'/home/danni/workspace/data/how2/data/orig/how2-300h-v1/data/train/text.id.en'

word_dict = dict()
sent_dict = dict()

with open(text_path, 'r') as f:
    for line in f.readlines():
        words = [pattern.sub('', i.strip()) for i in line.strip().lower().split(' ')[1:]]
        for word in words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
        sent = ' '.join(words)
        if sent not in sent_dict:
            sent_dict[sent] = 1
        else:
            sent_dict[sent] += 1

print('Vocab size:', len(word_dict))
print('Vocab count:')
for tup in sorted(word_dict.items(), key=lambda kv: kv[1], reverse=True)[:100]:
    print(tup)

print('Sent size:', len(sent_dict))
print('Sent count:')
for tup in sorted(sent_dict.items(), key=lambda kv: kv[1], reverse=True)[:100]:
    print(tup)
