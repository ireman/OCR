import pandas as pd
from imantics import Mask, Image, Category, Annotation
import cv2
import os
import numpy as np
from pathlib import Path
from COCO_ANNOTATION import load_images_from_folder
from collections import Counter

import torch
'''import torch.nn as nn 
import torch.optim as optim  
import torchvision.transforms as transforms
import torchvision

model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
#model.fc = torch.nn.Linear(num_ftrs, len(label_list))
#model.load_state_dict(torch.load(PATH))
model.load_state_dict(torch.load('cyrus_classifier.pt', map_location=torch.device('cpu')))
model.load_state_dict()'''
df = pd.read_csv(r'C:\Users\Home\Downloads\\final_csv.csv', skipinitialspace=True)
df['class'] = [1 for i,cls in enumerate(df['class'])]
df.to_csv('final_csv_no_label.csv', index=None, header=None, encoding='utf-8')
exit()
count_class = Counter(df['class'])
a = count_class.keys()
c = [i for i,cls in enumerate(a)]
b = pd.DataFrame()
b['label'] = a

b['number'] = c
b.to_csv('label_for_retinet_from_roi.csv', index=None, header=None, encoding='utf-8')
exit()
with open('texts.txt', 'r', encoding='utf-8') as infile:
    list_of_lines = [(line.strip()).split() for line in infile if
                     not (line.startswith('http') or line.startswith('\n') or len((line.strip()).split()) == 0)]
line_pos = 0
j = 35
file_name = 'cyrus_csv/{}.csv'
i = 0
image_list, image_list_coco = load_images_from_folder('cyrus2')
for img, image in zip(image_list, image_list_coco):
    df = pd.read_csv(file_name.format(j), skipinitialspace=True, header=None)
    j += 1
    #image = Image.from_path('cyrus_clean_img\Cyrus_page_36.png')
    #img = cv2.imread('cyrus_clean_img\Cyrus_page_36.png')

    '''bbox = df.keys()[1:5].astype(int)
    [X, Y, W, H] = bbox
    annotation = Annotation.from_bbox(bbox=[X, Y, X + W, Y + H],
                                          category=Category("Test")) #Category(curr_line[word_pos]))
    image.add(annotation)'''
    bbox = df.values[:, 1:5]
    sorted_array = np.argsort(bbox[:, 1])
    bbox_one_row = []
    bbox_all_rows = []
    curr_y = bbox[sorted_array[0]][1]

    for index in sorted_array:
        if abs(bbox[index][1]-curr_y) <= 25:
            bbox_one_row.append(bbox[index])
        else:
            bbox_one_row = np.vstack(bbox_one_row)
            sorted_one_row = bbox_one_row[np.argsort(bbox_one_row[:, 0])]
            bbox_all_rows.append(sorted_one_row)
            bbox_one_row = []
            curr_y = bbox[index][1]
            bbox_one_row.append(bbox[index])
    bbox_one_row = np.vstack(bbox_one_row)
    sorted_one_row = bbox_one_row[np.argsort(bbox_one_row[:, 0])]
    bbox_all_rows.append(sorted_one_row)
    print(image.file_name)
    line_in_page = 1
    for row, line in zip(bbox_all_rows, list_of_lines[line_pos: line_pos + len(bbox_all_rows)]):
        zip_row_line = list(zip(row, line))
        for bbox, label in zip_row_line:
            [X, Y, W, H] = bbox
            annotation = Annotation.from_bbox(bbox=[X, Y, X + W, Y + H],
                                              category=Category(label))  # Category(curr_line[word_pos]))
            image.add(annotation)
            #a = label.encode(encoding='UTF-8',errors='strict')
            Path('data_for_train_2\\' + label).mkdir(parents=True, exist_ok=True)
            #cv2.imwrite('data for train\\'+label+'\\'+str(i)+'.png', img[Y - 2:Y + H + 2, X:X + W])
            #cv2.imwrite('data for train1\\'+str(i)+'.png', img[Y - 2:Y + H + 2, X:X + W])
            destination = 'data_for_train_2\\'+label+'\\'
            gel = 'p_' +str(j-1)+'_line_'+str(line_in_page)+'_sample_'+ str(i)+'.png'
            script_path = os.getcwd()
            os.chdir(destination)
            cv2.imwrite(gel, img[Y - 2:Y + H + 2, X:X + W])
            os.chdir(script_path)

            i += 1
        line_in_page += 1
    line_pos += len(bbox_all_rows)
    coco_json = image.export(style='coco')
    image.save('Cyrus_tiny\\annotation_' + image.file_name + '_.json', style='coco')