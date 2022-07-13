import cv2
import numpy as np
from FOSSEY_OCR import load_images_from_folder

image_list = load_images_from_folder('fossy_from_shai')
origin_image_list = load_images_from_folder('fossy_from_shai')

for img in image_list:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    imS = cv2.resize(thresh, (560, 680))

    cv2.imshow('result', imS)
    cv2.waitKey()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 150:
            cv2.drawContours(opening, [c], -1, (0,0,0), -1)

    result = 255 - opening
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('opening', opening)
    imS = cv2.resize(result, (560, 680))

    cv2.imshow('result', imS)
    cv2.waitKey()