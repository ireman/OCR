import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

image_list = load_images_from_folder('Cyrus_tiny')
for image in image_list:
    #image = cv2.imread('Cyrus_tiny\Cyrus_page_161.png')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    zero_line = np.zeros(20)
    #horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    horizontal_kernel = np.array([[1, 1, 1],
                       [0, 0, 0],
                        [0, 0, 0],[0, 0, 0],
                       [1, 1, 1]], dtype=np.uint8)
    horizontal_kernel = np.repeat(horizontal_kernel, repeats=20, axis=1)
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        [X, Y, W, H] = cv2.boundingRect(c)
        cv2.rectangle(image, (X - 350, Y - 25), (X + 450, Y), (255, 255, 255), -1)
        #cv2.drawContours(image, [c], -1, (255,255,255), 2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines += cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (0, 0, 255), 2)

    # Repair image
    #repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    #result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    #cv2.imshow('thresh', thresh)
    #cv2.imshow('detected_lines', cv2.resize(detected_lines, (960, 740)))
    cv2.imshow('image', cv2.resize(image, (960, 740)))
    #cv2.imshow('result', cv2.resize(image, (960, 740)))
    cv2.waitKey()