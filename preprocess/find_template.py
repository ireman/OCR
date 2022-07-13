# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
'''from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

import cv2

img = cv2.imread(r'pic.png')
#cv2.imshow('Image', img)

#--- create a blank image of the same size for storing the green rectangles (boundaries) ---
black = np.zeros_like(img)

#--- convert your image to grayscale and apply a threshold ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#--- perform morphological operation to ensure smaller portions are part of a single character ---
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
threshed = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

#--- find contours ---
Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for contour in Contours:

    #--- select contours above a certain area ---
    if cv2.contourArea(contour) > 200:

        #--- store the coordinates of the bounding boxes ---
        [X, Y, W, H] = cv2.boundingRect(contour)

        #--- draw those bounding boxes in the actual image as well as the plain blank image ---
        cv2.rectangle(img, (X, Y), (X + W, Y + H), (0,0,255), 2)
        cv2.rectangle(black, (X, Y), (X + W, Y + H), (0,255,0), 2)

cv2.imshow('contour', img)
cv2.waitKey()
cv2.imshow('black', black)
cv2.waitKey()'''
#from draw_bb import remove_diag_line
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
import pandas as pd
all_pred = pd.read_csv(r'C:\Users\Home\PycharmProjects\OCR\cuneiform_ocr\all_pred_for_retinanet.csv', keep_default_na=False,encoding='utf-8')
forbiden_page = [13, 19, 25, 35, 36, 37, 40, 41, 44, 47, 48, 49, 52, 55, 56, 59, 60, 63, 66, 72, 75, 76, 77, 87, 88, 92, 96, 97, 101, 104, 107, 110, 118, 121, 122, 123, 124, 131, 132, 135, 142, 143, 147, 149, 151, 152, 153, 174, 184, 190, 195, 206, 209, 211, 213, 215, 227, 229, 239, 242, 255, 261, 277, 282, 286, 291, 298, 299, 301, 303, 311, 313, 314, 331, 334, 344, 349, 357, 363, 367, 382]
dfnew = all_pred.groupby('image').filter(lambda x: x['image'].split('\\')[1].split('.')[0] not in forbiden_page)
a = 5
# construct the argument parser and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image where we'll apply template matching")
ap.add_argument("-t", "--template", type=str, required=True,
	help="path to template image")
ap.add_argument("-b", "--threshold", type=float, default=0.8,
	help="threshold for multi-template matching")
args = vars(ap.parse_args())# load the input image and template image from disk, then grab the
# template image spatial dimensions'''
print("[INFO] loading images...")
image = cv2.imread(r'Cyrus\Cyrus-page-035.jpg')
cv2.imshow("Image", image)
#image = remove_diag_line(image)
template = cv2.imread(r'crop_images\46.jpg')
(tH, tW) = template.shape[:2]
# display the  image and template to our screen
cv2.imshow("Template", template)
# convert both the image and template to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# perform template matching
print("[INFO] performing template matching...")
result = cv2.matchTemplate(imageGray, templateGray,
	cv2.TM_CCOEFF_NORMED)
# find all locations in the result map where the matched value is
# greater than the threshold, then clone our original image so we
# can draw on it
(yCoords, xCoords) = np.where(result >= 0.50)
clone = image.copy()
print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))
# loop over our starting (x, y)-coordinates
for (x, y) in zip(xCoords, yCoords):
	# draw the bounding box on the image
	cv2.rectangle(clone, (x, y), (x + tW, y + tH),
		(255, 0, 0), 3)
# show our output image *before* applying non-maxima suppression
#cv2.imshow("Before NMS", clone)
#cv2.waitKey(0)
# initialize our list of rectangles
rects = []
# loop over the starting (x, y)-coordinates again
for (x, y) in zip(xCoords, yCoords):
	# update our list of rectangles
	rects.append((x, y, x + tW, y + tH))
# apply non-maxima suppression to the rectangles
pick = non_max_suppression(np.array(rects))
print("[INFO] {} matched locations *after* NMS".format(len(pick)))
# loop over the final bounding boxes
for (startX, startY, endX, endY) in pick:
	# draw the bounding box on the image
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(255, 0, 0), 3)
# show the output image
cv2.imshow("After NMS", image)#cv2.resize(image, (760, 780)))
cv2.waitKey(0)