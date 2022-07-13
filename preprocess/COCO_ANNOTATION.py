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
        image_coco = Image.from_path(folder + "\\" +  filename)
        if img is not None:
            images.append(img)
            images_for_coco.append(image_coco)
    return images, images_for_coco



if __name__ == '__main__':
    image_list, image_list_coco = load_images_from_folder('separateCyrusImg\images')
    page_num = 1
    coco_json = {}
    for img, image in zip(image_list,image_list_coco):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        src_img = img.copy()

        ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # --- perform morphological operation to ensure smaller portions are part of a single character ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        threshed = cv2.morphologyEx(th2, cv2.MORPH_GRADIENT, kernel)
        #cv2.imshow('threshed', threshed)

        # --- find contours ---
        Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        #Contours.sort(key=lambda x: get_contour_precedence(x, img.shape[1]))
        x_vals = []
        y_vals = []
        w_vals = []
        h_vals = []
        for contour in Contours:
            [X, Y, W, H] = cv2.boundingRect(contour)

            area = cv2.contourArea(contour)
            if 8 < W < 100 and 8 < H < 50 and cv2.contourArea(contour) > 60:
                # --- draw those bounding boxes in the actual image as well as the plain blank image ---
                crop_img = gray[Y - 1:Y + H + 1, X - 1:X + W + 1]
                zero_pixel = (np.sum(crop_img == 0)/crop_img.size)*100
                if zero_pixel >= 12:
                    cv2.rectangle(src_img, (X - 1, Y - 1), (X + W + 1, Y + H + 1), (0, 0, 255), 2)
                    x_vals.append(X - 1)
                    y_vals.append(Y - 1)
                    w_vals.append(X + W + 1)
                    h_vals.append(Y + H + 1)
                    #cv2.imshow('crop', crop_img)
                    #cv2.waitKey()
                annotation = Annotation.from_bbox(bbox=[X - 1, Y - 1, X + 1 + W, Y + 1 + H],
                                                  category=Category("Test"))#Category(curr_line[word_pos]))
                image.add(annotation)
        #cv2.imshow('result', src_img)
        #cv2.waitKey()
        data = {'X': x_vals, 'Y': y_vals, 'W': w_vals, 'H': h_vals}
        df = pd.DataFrame(data)
        df1 = df.sort_values(by=['Y'])
        bbox = df1.values[:, 0:4]
        y_val = bbox[0][1]
        line_index = []
        line_num = 1
        for k, box in enumerate(bbox):
            y_curr = box[1]
            if abs(y_val-y_curr)<=25:
                line_index.append(line_num)
            else:
                y_val = y_curr
                line_num+=1
                line_index.append(line_num)
        df1["lineNumber"] = line_index
        bbox = df1.values[:, 0:4]
        sorted_array = np.argsort(bbox[:, 1])
        bbox_one_row = []
        bbox_all_rows = []
        curr_y = bbox[sorted_array[0]][1]

        for index in sorted_array:
            if abs(bbox[index][1] - curr_y) <= 25:
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
        bbox_all_rows_np = np.array(bbox_all_rows)
        df = pd.DataFrame.from_records(bbox_all_rows)
        file_name = image.file_name.split('.')[0]
        df.to_csv(path_or_buf='separateCyrusImg\\csv\\'+file_name+'.csv', index=None, header=True)

        coco_json1 = image.export(style='coco')

        if len(coco_json):
            coco_json['categories'].append(coco_json1['categories'])
            coco_json['annotations'].append(coco_json1['annotations'])
            coco_json['images'].append(coco_json1['images'])
        else:
            coco_json = coco_json1
        # Saves to file
        #cv2.imshow('result', src_img)
        #cv2.waitKey()
        #image.save('separateCyrusImg\\coco\\annotation_page_'+file_name+'.json', style='coco')
        page_num += 1

