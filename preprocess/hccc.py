from os import listdir
import pandas as pd
from os.path import isfile, join
import numpy as np
import os
import cv2
import pandas as pd
from scipy import stats
from collections import Counter
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder,filename)
        stream = open(os.path.join(path), "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread(os.path.join(path,img))
        if bgrImage is not None:
            images.append((bgrImage,filename))
    return images

mypath=r'C:\Users\Home\Downloads\consensus_data13'
onlyfiles = [f for f in listdir(mypath)]
df = pd.read_csv('img_name_label_filtered.csv', skipinitialspace=True)
train, test = train_test_split(df, test_size=0.2)

a = Counter(df['label'])
labels_name = a.keys()
df = pd.DataFrame()
df['label'] = onlyfiles
df = pd.DataFrame()
df['label'] = onlyfiles
df.to_csv(path_or_buf='filtered_label.csv', index=None, header=True, encoding='utf-8')
a = 5

df_all_name_label = pd.DataFrame()
for file in onlyfiles:
    hight_list = []
    width_list = []
    names = []
    path = r'C:\Users\Home\Downloads\consensus_data13\\' + file
    image_list = load_images_from_folder(path)
    for img, img_name in image_list:
        height, width = img.shape
        hight_list.append(height)
        width_list.append(width)
        names.append(img_name)
    df = pd.DataFrame()
    df['height'] = hight_list
    df['width'] = width_list
    df['names'] = names
    df['label'] = file
    if len(df['label']) > 5:
        df = df[(np.abs(stats.zscore(df[['height', 'width']])) < 1.3).all(axis=1)]
    #describe_height = df['height'].describe()
    #describe_width = df['width'].describe()
    df_all_name_label = pd.concat([df_all_name_label,df[['names', 'label']]])
df_all_name_label.to_csv(path_or_buf='img_name_label_filtered.csv', index=None, header=True, encoding='utf-8')
