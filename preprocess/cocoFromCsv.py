import pandas as pd
from imantics import Mask, Image, Category, Annotation
import cv2
import os

df = pd.read_csv('cyrus_csv\\39.csv', skipinitialspace=True)
image = Image.from_path('cyrus2\Cyrus-page-39.png')
bbox = df.keys()[1:5].astype(int)
[X, Y, W, H] = bbox
annotation = Annotation.from_bbox(bbox=[X, Y, X + W, Y + H],
                                      category=Category("Test")) #Category(curr_line[word_pos]))
image.add(annotation)

for bbox in df.values[:, 1:5]:
    [X, Y, W, H] = bbox
    annotation = Annotation.from_bbox(bbox=[X, Y, X + W, Y + H],
                                      category=Category("Test")) #Category(curr_line[word_pos]))
    image.add(annotation)

coco_json = image.export(style='coco')
image.save('Cyrus_coco\\coco_from_csv'+image.file_name+'.json', style='coco')