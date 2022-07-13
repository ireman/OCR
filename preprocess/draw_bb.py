import cv2
import os
import numpy as np
import pandas as pd
from imantics import Mask, Image, Category, Annotation

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.4
fontColor              = (0, 0, 255)
lineType               = 2

def get_contour_precedence(contour, cols):
    tolerance_factor = 20
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]



def load_images_from_folder(folder):
    images = []
    images_for_coco = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def remove_lines(image):
    #image = cv2.imread('pic.PNG')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
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


def remove_horizon_lines(image):
    #image = cv2.imread('pic.PNG')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    return cv2.merge([image, image, image])

if __name__ == '__main__':
    #image_list, image_list_coco = load_images_from_folder('Cyrus')
    #origin_image_list = load_images_from_folder('Cyrus')
    #k = 0
    with open('StrassmaierTransliteration.txt', 'r', encoding='utf-8') as infile:
        list_of_lines = [(line.strip()).split() for line in infile if
                         not (line.startswith('http') or line.startswith('\n'))]
    line_pos = 0
    for img, image in zip(image_list,image_list_coco):
        src_img = img.copy()
        #cv2.imshow('origin', img)origin_image_list[0]
        #clean_line_img = remove_horizon_lines(img)
        clean_img = remove_lines(img)
        #cv2.imshow('clean',clean_img)
        gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
        #img = cv2.imread('result4.jpg', 0)
        '''cv2.threshold(255-gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,gray)
        #img_origin = cv2.imread(r'pic.PNG')

        contours, hier = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a white rectangle to visualize the bounding rect
            if w < 80 and h < 40 and cv2.contourArea(c) > 15:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)

        #cv2.drawContours(img_origin, contours, -1, (255, 255, 0), 2)

        cv2.imshow("output.png",img)'''
        ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imshow('th2', th2)

        # --- perform morphological operation to ensure smaller portions are part of a single character ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        threshed = cv2.morphologyEx(th2, cv2.MORPH_GRADIENT, kernel)
        #cv2.imshow('threshed', threshed)

        # --- find contours ---
        Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        '''for contour in Contours:

            #--- select contours above a certain area ---
            if (cv2.contourArea(contour) > 80) and (cv2.contourArea(contour) < 800):

                #--- store the coordinates of the bounding boxes ---
                [X, Y, W, H] = cv2.boundingRect(contour)

                #--- draw those bounding boxes in the actual image as well as the plain blank image ---
                cv2.rectangle(img_origin, (X, Y), (X + W, Y + H), (0,0,255), 2)
                cv2.rectangle(black, (X, Y), (X + W, Y + H), (0,255,0), 2)'''
        x_vals = []
        y_vals = []
        w_vals =[]
        h_vals = []
        trans_vals = []


        Contours.sort(key=lambda x: get_contour_precedence(x, img.shape[1]))
        prev_x_pos = 0
        word_pos = 0
        curr_line = list_of_lines[line_pos]
        i = 0
        #image = Image.from_path('Cyrus\Cyrus-page-035.jpg')
        for contour in Contours:
            [X, Y, W, H] = cv2.boundingRect(contour)

            area = cv2.contourArea(contour)
            if 8 < W < 80 and 8 < H < 40 and cv2.contourArea(contour) > 40:
                # --- draw those bounding boxes in the actual image as well as the plain blank image ---
                crop_img = img[Y - 2:Y + H + 2, X:X + W]
                #cv2.imwrite('crop_img.png', crop_img)

                cv2.imshow("cropped", crop_img)
                #cv2.waitKey(0)
                '''cv2.rectangle(origin_image_list[k], (X - 1, Y - 1), (X + W + 1, Y + H + 1), (0, 0, 255), 2)
                x_vals.append(X - 1)
                y_vals.append(Y - 1)
                w_vals.append(X + W + 1)
                h_vals.append(Y + H + 1)'''

                key = cv2.waitKey(0)
                while (key != 'y' and key != 'n'):
                    if (key == ord('y')):
                        #cv2.imwrite('crop_images\{0}.jpg'.format(i), crop_img)
                        #i += 1
                        if X > prev_x_pos:
                            prev_x_pos = X
                        else:
                            # line end, start new line
                            line_pos += 1
                            curr_line = list_of_lines[line_pos]
                            prev_x_pos = X
                            word_pos = 0
                        line_len = len(curr_line)
                        #skip unneeded words
                        while (line_len > word_pos) and (curr_line[word_pos] == '<UNK>'
                                    or curr_line[word_pos] == '<BRK>' or curr_line[word_pos] == 'o'):
                            word_pos += 1
                        if line_len <= word_pos:
                            list_of_lines[line_pos].append('miss')
                        '''cv2.putText(origin_image_list[k], curr_line[word_pos],
                                    (X, Y),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                        cv2.imshow("TEXT ON IMG", origin_image_list[k])
                        cv2.waitKey(0)'''
                        cv2.rectangle(src_img, (X - 1, Y - 1), (X + W + 1, Y + H + 1), (0, 0, 255), 2)
                        x_vals.append(X - 1)
                        y_vals.append(Y - 2)
                        w_vals.append(X + W + 1)
                        h_vals.append(Y + H + 2)
                        trans_vals.append(curr_line[word_pos])
                        annotation = Annotation.from_bbox(bbox=[X - 1, Y - 1, X + 1 + W, Y + 1 + H],
                                                          category=None)#Category(curr_line[word_pos]))
                        image.add(annotation)
                        cv2.destroyAllWindows()
                        #cv2.imwrite(curr_line[word_pos].encode('utf-8') + '.png', img[Y-2:Y + H + 2, X:X + W])
                        word_pos += 1
                        break
                    elif key == ord('n'):
                        cv2.destroyAllWindows()
                        break


        '''min_y_in_line = y_vals[0]
        for i in range(len(y_vals)):
            if y_vals[i] - min_y_in_line <= 5:
                y_vals[i] = min_y_in_line
            else:
                #line end, start new line
                min_y_in_line = y_vals[i]'''

        data = {'ImageID': 1, 'Source': 1, 'Confidence': 1, 'LabelName': trans_vals, 'XMin': x_vals,'XMax': w_vals,
                'YMin': y_vals, 'YMax': h_vals}
        df = pd.DataFrame(data)
        df.to_csv(path_or_buf='output_csv_file_sort.csv', index=None, header=True)
        cv2.imwrite('result.png',src_img)
        cv2.imshow('bounding boxes', src_img)
        cv2.waitKey(0)
        #k += 1
        coco_json = image.export(style='coco')
        # Saves to file
        image.save('annotation.json', style='coco')


