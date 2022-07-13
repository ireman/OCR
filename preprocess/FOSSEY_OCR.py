import cv2
import os
import numpy as np
import pandas as pd
import pytesseract
import unidecode
from itertools import groupby
from operator import itemgetter
from pathlib import Path


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 3
fontColor              = (0, 0, 255)
lineType               = 2

def get_contour_precedence(contour, cols):
    tolerance_factor = 100
    origin = cv2.boundingRect(contour)
    return ((origin[0] // tolerance_factor) * tolerance_factor) * cols


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def first_nonzero(arr, axis, invalid_val=10000):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def remove_diag_line(image):
    #image = cv2.imread('pic.PNG')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        [X, Y, W, H] = cv2.boundingRect(c)
        cv2.rectangle(image, (X - 10, Y-5), (X + W+5, Y + H+10), (255, 255, 255), -1)
        #cv2.drawContours(image, [c], -1, (255, 255, 255), -1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    index = first_nonzero(detected_lines, 1)
    #cv2.imshow('lines', detected_lines)
    #cv2.waitKey()
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        [X, Y, W, H] = cv2.boundingRect(c)
        cv2.rectangle(image, (X - 10, Y-5), (X + W+10, Y + H+5), (255, 255, 255), -1)
        #cv2.drawContours(image, [c], -1, (255, 255, 255), -1)

    # Bitwise-xor with original image
    #cv2.imshow('image', image)
    #cv2.waitKey()
    thresh = cv2.merge([thresh, thresh, thresh])

    result = cv2.bitwise_xor(255-thresh, image)
    return image


def remove_horizon_lines(image):
    #image = cv2.imread('pic.PNG')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    return cv2.merge([image, image, image])

if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image_list = load_images_from_folder('fossey\\fossy_from_shai')
    origin_image_list = load_images_from_folder('fossey\\fossy_from_shai')
    k = 0
    ii = 0

    for img in image_list:
        #cv2.imshow('clean', cv2.resize(clean_img, (560, 680)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 150:
                cv2.drawContours(opening, [c], -1, (0, 0, 0), -1)

        result = 255 - opening
        result = cv2.merge([result, result, result])
        gray = result
        gray = remove_diag_line(gray)

        imS = cv2.resize(gray, (560, 680))
        #cv2.imshow('result', imS)
        #cv2.waitKey()
        # Define the structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 2))
        dilation = cv2.dilate(gray, kernel, iterations=1)

        gradient1 = dilation
        imS = cv2.resize(gradient1, (960, 740))
        #cv2.imshow('gradient1', imS)
        #cv2.waitKey()
        # Apply the opening operation
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        #cv2.imshow('opening', cv2.resize(opening, (560, 680)))
        # Apply the closing operation
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow('closing', cv2.resize(closing, (560, 680)))
        # Approach-1: Perform erosion and dilation separately and then subtract
        erosion = cv2.erode(gray, kernel, iterations=1)
        #cv2.imshow('erosion', cv2.resize(erosion, (560, 680)))

        gradient1 = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('Manuel-1-41\page' + str(k) + 'OCR.jpg', origin_image_list[k])

        # Approach-2: Use cv2.morphologyEx()
        #gradient1 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        th2 = cv2.threshold(gradient1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #cv2.imshow('th2', cv2.resize(th2, (560, 680)))

        # --- perform morphological operation to ensure smaller portions are part of a single character ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        threshed = cv2.morphologyEx(th2, cv2.MORPH_GRADIENT, kernel,iterations=1)
        #cv2.imshow('threshed', threshed)
        # --- find contours ---
        Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        Contours.sort(key=lambda x: get_contour_precedence(x, img.shape[2]))
        j = 0
        #list_of_col = [cv2.boundingRect(contour) for contour in Contours if 380 < cv2.contourArea(contour) < 8000]
        
        height, width, channels = img.shape
        y_pos = height
        curr_col = []
        all_col = []
        for contour in Contours:
            [X, Y, W, H] = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if 20 < W and 15 < H and 180 < area:
                if y_pos >= Y - H:
                    y_pos = Y - H
                    curr_col.append([X, Y-15, W, H+15])
                    crop_img = gray[Y-3:Y + H+3, X:X + W]
                else:
                    all_col.append(curr_col)
                    curr_col = []
                    crop_img = gray[Y-3:Y + H+3, X:X + W]
                    if len(all_col) == 2:
                        break;
                    curr_col.append([X, Y-15, W, H+15])
                    y_pos = height
                # --- draw those bounding boxes in the actual image

                cv2.rectangle(origin_image_list[k], (X, Y - 15), (X + W, Y + H + 15), (0, 0, 255), 2)
                cv2.putText(origin_image_list[k], str(j),
                            (X, Y),
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                j += 1

                crop_img = gray[Y:Y + H, X:X + W]
                #text = pytesseract.image_to_string(crop_img, config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
                #print(text)
        imS = cv2.resize(origin_image_list[k], (560, 680))
        cv2.imshow('canny', imS)
        cv2.waitKey(0)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        '''for crop_number in all_col[0]:
            [X, Y, W, H] = crop_number
            ''''''crop_img1 = origin_image_list[k][Y-7:Y + H+7, X:X + W]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 3))

            threshed = cv2.morphologyEx(th2[Y-7:Y + H+7, X:X + W], cv2.MORPH_GRADIENT, kernel, iterations=1)
            gray = cv2.cvtColor(crop_img1, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]''''''
            #gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blurred = cv2.GaussianBlur(gray[Y-7:Y + H+7, X:X + W], (9, 9), 0)
            #cv2.imshow('blurred', blurred)
            canny = cv2.Canny(blurred, 120, 255, 1)
            kernel = np.ones((3, 3), np.uint8)
            dilate = cv2.dilate(canny, kernel, iterations=1)
            cv2.imshow('canny', dilate)
            cv2.waitKey(0)
            Contours_numbers, Hierarchy1 = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #cv2.imshow('threshed', threshed)
            for contour1 in Contours_numbers:
                area = cv2.contourArea(contour1)
                if 8 < W and 8 < H and 500 < area:
                    [X1, Y1, W1, H1] = cv2.boundingRect(contour1)
                    candidate_num = blurred[Y1-7:Y1 + H1+7, X1:X1 + W1]
                    kernel = np.ones((5, 5), np.uint8)
                    threshed1 = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel, iterations=1)
                    ''''''cv2.imshow('blurred', candidate_num)
                    #big_image = cv2.resize(candidate_num, (200, 200), fx=7.5, fy=7.5)
                    #cv2.imshow('big_image', big_image)
                    cv2.waitKey(0)''''''
                    ocr_as_str = pytesseract.image_to_string(candidate_num,
                                                             config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
                    #cv2.rectangle(blurred, (X1, Y1), (X1 + W1, Y1 + H1), (0, 0, 255), 2)

                    print(ocr_as_str)
                    cv2.imwrite('fossey_numbers\\'+str(ii)+'.png', candidate_num)
                    ii += 1
                    #cv2.ellipse(crop_img1, (X1, Y1), (X1 + W1, Y1 + H1), (0, 0, 255), 2)
            imS = cv2.resize(origin_image_list[k], (560, 680))
            #cv2.imshow('bounding boxes', blurred)
            #cv2.imshow('crop', origin_image_list[k])
            #cv2.waitKey(0)
            # cv2.imshow('threshed', threshed)
            # --- find contours ---
            Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)'''
        zip_list = list(zip(all_col[0], all_col[1]))
        ocr_as_str = [pytesseract.image_to_string(cv2.GaussianBlur(gray[Y-7:Y + H+7, X:X + W], (3, 3), 0), config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
                        for [X, Y, W, H], _ in zip_list]
        ocr_as_str = [num.strip() for num in ocr_as_str]
        #ocr_num = [int(element) for element in ocr_as_str]
        ocr_as_int = []
        for element in ocr_as_str:
            try:
                ocr_as_int.append(int(element))
            except:
                ocr_as_int.append(-1)

        for f, g in groupby(enumerate(ocr_as_int), lambda ix : ix[0] - ix[1]):
            serial_number = list(map(itemgetter(1), g))
        x_end_pos = width
        #text = [pytesseract.image_to_string(img[Y:Y + H, X:X + width], lang='eng+fra') for i, [X, Y, W, H] in enumerate(all_col[1])]
        Path('fossey' + str(k)).mkdir(parents=True, exist_ok=True)
        path = 'fossey' + str(k)+'\\'
        # cv2.imwrite('data for train\\'+label+'\\'+str(i)+'.png', img[Y - 2:Y + H + 2, X:X + W])
        # cv2.imwrite('data for train1\\'+str(i)+'.png', img[Y - 2:Y + H + 2, X:X + W])

        [cv2.imwrite(path+str(ocr_as_int[i])+'.png', gray[Y-2:Y + H+2, X-2:X + W+2]) for i, (_, [X, Y, W, H]) in enumerate(zip_list)]
        #cv2.imwrite('Manuel-1-41\page' + str(k) + 'OCR.jpg', origin_image_list[k])
        '''imS = cv2.resize(origin_image_list[k], (560, 680))
        cv2.imshow('bounding boxes', imS)
        imS = cv2.resize(gray, (560, 680))

        cv2.imshow('gray', imS)'''

        #cv2.waitKey(0)
        k += 1


