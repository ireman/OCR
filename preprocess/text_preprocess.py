import cv2
import numpy as np
import pandas as pd
from collections import Counter
from os import listdir
from os.path import isfile, join

#onlyfiles = sorted([int(f.split('.')[0]) for f in listdir('separateCambysesImg\\') if isfile(join('separateCambysesImg\\', f))])

with open('Cambyse_texts_unicode.txt', 'r', encoding='utf-8') as infile:
    list_of_lines = [(" ".join(line.split())).split('\n') for line in infile if
                     not (line.startswith('\n') or len((line.strip()).split()) == 0)]# or line.startswith('http'))]
    count = 0
    for listElem in list_of_lines:
        count += len(listElem)
    a = Counter(x for xs in list_of_lines for x in set(xs)).most_common()
    a.reverse()
    keys = []
    label = []
    for item in a:
        keys.append(item[0])
        label.append(item[1])
    data = {'label': keys}
    df = pd.DataFrame(data)
    #df.to_csv(path_or_buf='labels_with_unicode4.csv', index=None, header=True, encoding='utf-8')
    #exit()

http_flag = False
curr_text = ''
i = 1
for line in list_of_lines[1:]:
    if line[0].startswith('http'):
        with open('CambyseSeparateTxt\\'+str(i)+'.txt', 'w', encoding='utf-8') as f:
            f.write(curr_text)
            curr_text = ''
            i += 1
        continue
    curr_text += line[0]+'\n'

with open('CambyseSeparateTxt\\' + str(i) + '.txt', 'w', encoding='utf-8') as f:
    f.write(curr_text)