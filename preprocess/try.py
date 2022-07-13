import cv2
import numpy as np
import os
from imantics import Mask, Image, Category, Annotation


def remove_diag_line(image):
    #image = cv2.imread('pic.PNG')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    a = np.array([1, 1])

    kernel = np.array(np.flip(np.diag(a), 1), dtype=np.uint8)
    kernel = np.repeat(kernel, repeats=2, axis=0)

    # Create diagonal kernel
    '''kernel = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0]], dtype=np.uint8)'''
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel, iterations=1)
    # Find contours and filter using contour area to remove noise
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    # Bitwise-xor with original image
    thresh = cv2.merge([thresh, thresh, thresh])

    result = cv2.bitwise_xor(thresh, image)
    # result=255-thresh-result
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('opening', opening)
    #cv2.imshow('result', 255 - result)
    #cv2.imwrite("result4.jpg", 255 - result)

    #cv2.waitKey()
    #return cv2.merge([255 - result, 255 - result, 255 - result])
    return result


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

if __name__ == '__main__':
    image_list = load_images_from_folder('Cyrus')
    for img in image_list:
        #img = cv2.imread(r'Cyrus\Cyrus-page-036.jpg')
        #cv2.imshow('Image', img)
        img_origin = img.copy()
        #--- create a blank image of the same size for storing the green rectangles (boundaries) ---
        #black = np.zeros_like(img)
        #img = remove_diag_line(img)
        #--- convert your image to grayscale and apply a threshold ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imshow('th2', th2)

        #--- perform morphological operation to ensure smaller portions are part of a single character ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 80))
        threshed = cv2.morphologyEx(th2, cv2.MORPH_GRADIENT, kernel)
        #cv2.imshow('threshed', threshed)
        template = cv2.imread(r'crop_images\142.jpg')

        #--- find contours ---
        Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        '''for contour in Contours:
        
            #--- select contours above a certain area ---
            if (cv2.contourArea(contour) > 80) and (cv2.contourArea(contour) < 800):
        
                #--- store the coordinates of the bounding boxes ---
                [X, Y, W, H] = cv2.boundingRect(contour)
        
                #--- draw those bounding boxes in the actual image as well as the plain blank image ---
                cv2.rectangle(img_origin, (X, Y), (X + W, Y + H), (0,0,255), 2)
                cv2.rectangle(black, (X, Y), (X + W, Y + H), (0,255,0), 2)'''
        i = 0
        annotation_list = []
        image = Image.from_path('Cyrus\Cyrus-page-035.jpg')

        for contour in Contours:
            [X, Y, W, H] = cv2.boundingRect(contour)
            if W*H > 100000:
            #--- draw those bounding boxes in the actual image as well as the plain blank image ---

                #cv2.rectangle(black, (X, Y), (X + W, Y + H), (0,255,0), 2)
                '''if i<100:
                    crop_img = img_origin[Y:Y + H, X:X + W]
                    cv2.imwrite("crop_images\cropped%i.png"%i, crop_img)
                    i+=1'''
                cv2.rectangle(img_origin, (X+50, Y+90), (X-20 + W, Y-65 + H), (0,0,255), 2)
                annotation_list.append([X-1, Y-1, X+1 + W, Y+1 + H])
                annotation = Annotation.from_bbox(bbox=[X-1, Y-1, X+1 + W, Y+1 + H], category=Category("Category Name"))
                image.add(annotation)

        cv2.imshow('contour', img_origin)

        #cv2.imshow('black', black)
        #cv2.imwrite('contour.png', img)
        cv2.waitKey(0)

        # dict of coco
        coco_json = image.export(style='coco')
        # Saves to file
        #image.save('annotation.json', style='coco')
        cv2.waitKey()