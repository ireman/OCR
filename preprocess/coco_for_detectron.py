import numpy as np
import json
import pandas as pd
import cv2
import os


def load_images_from_folder(folder):
    images = []
    images_name = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            images_name.append(filename)
    return images, images_name

p_dict = {}
path = 'img_name_label_all_pred.csv'
save_json_path = 'all_pred_traincoco_1_label.json'
page = [13, 19, 25, 35, 36, 37, 40, 41, 44, 47, 48, 49, 52, 55, 56, 59, 60, 63, 66, 72, 75, 76, 77, 87, 88, 92, 96, 97, 101, 104, 107, 110, 118, 121, 122, 123, 124, 131, 132, 135, 142, 143, 147, 149, 151, 152, 153, 174, 184, 190, 195, 206, 209, 211, 213, 215, 227, 229, 239, 242, 255, 261, 277, 282, 286, 291, 298, 299, 301, 303, 311, 313, 314, 331, 334, 344, 349, 357, 363, 367, 382]
images_list, img_name = load_images_from_folder('separateCyrusImg\images')
for im, p in zip(images_list, img_name):
    w = im.shape[1]
    h = im.shape[0]
    p_dict[p] = (w, h)
'''for p in range(1,385):
    p = str(p) + '.png'
    im = cv2.imread('separateCyrusImg\images\\'+str(p))
    w = im.shape[1]
    h = im.shape[0]
    p_dict[p] = (w,h)'''
data = pd.read_csv(path,encoding='utf-8')
height = []
width = []

for img in data.filename:
    (w,h) = p_dict[img.split('\\')[-1]]
    height.append(h)
    width.append(w)
data['height'] = height
data['width'] = width
data['label'] = ['sign']*len(width)

images = []
categories = []
annotations = []

category = {}
category["supercategory"] = 'none'
category["id"] = 0
category["name"] = 'None'
categories.append(category)
data['filename'] = data['filename'].str.split('\\')
data['fileid'] = data['filename'].str.split('\\')[-1].astype('category').cat.codes
data['categoryid'] = pd.Categorical(data['label'],ordered= True).codes
data['categoryid'] = data['categoryid']+1
data['annid'] = data.index

def image(row):
    image = {}
    image["height"] = row.height
    image["width"] = row.width
    image["id"] = row.fileid
    image["file_name"] = row.filename.split('\\')[-1]
    return image

def category(row):
    category = {}
    category["supercategory"] = 'None'
    category["id"] = row.categoryid
    category["name"] = row[6]
    return category

def annotation(row):
    annotation = {}
    area = (row.xmax -row.xmin)*(row.ymax - row.ymin)
    annotation["segmentation"] = []
    annotation["iscrowd"] = 0
    annotation["area"] = area
    annotation["image_id"] = row.fileid

    annotation["bbox"] = [row.xmin, row.ymin, row.xmax -row.xmin,row.ymax-row.ymin ]

    annotation["category_id"] = row.categoryid
    annotation["id"] = row.annid
    return annotation

for row in data.itertuples():
    annotations.append(annotation(row))

imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
for row in imagedf.itertuples():
    images.append(image(row))

catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
for row in catdf.itertuples():
    categories.append(category(row))

data_coco = {}
data_coco["images"] = images
data_coco["categories"] = categories
data_coco["annotations"] = annotations


json.dump(data_coco, open(save_json_path, "w"), indent=4)