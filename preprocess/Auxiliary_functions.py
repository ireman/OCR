import cv2
import pandas as pd
import numpy as np

def union(box1,box2):
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = max(box1[2], box2[2])
    h = max(box1[3], box2[3])
    return x, y, w, h


def intersection(box1,box2):
    if 2 < box1[2]-box2[0] < 50 and abs(box1[3]-box2[3])<20:
        return True
    return False


def combine_boxes(boxes):
    new_array = []
    i = 0
    while i < len(boxes)-1:
        if intersection(boxes[i], boxes[i+1]):
            new_array.append(union(boxes[i], boxes[i+1]))
            i+=2
        else:
            new_array.append(boxes[i])
            i+=1
    if i <=len(boxes)-1:
        new_array.append(boxes[i])
    return np.array(new_array).astype('int')

def get_boundingBox_of_img(self, img: np.ndarray, is_clean: bool = 1) -> pd.DataFrame:
    """
    this function get an image of cuneiform hand-copy
    and return a dataFrame with bounding box for each cuneiform sign
    """
    if not is_clean:
        """
        clean noise from image
        """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_img = img.copy()
    src_img1 = img.copy()


    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- perform morphological operation to ensure smaller portions are part of a single character ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    threshed = cv2.morphologyEx(th2, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow('threshed', threshed)

    # --- find contours ---
    Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Contours.sort(key=lambda x: self.get_contour_precedence(x, img.shape[1]))
    x_vals = []
    y_vals = []
    w_vals = []
    h_vals = []
    all_b = []
    for contour in Contours:
        [X, Y, W, H] = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)
        if 8 < W < 130 and 8 < H < 50 and cv2.contourArea(contour) > 60:
            # --- draw those bounding boxes in the actual image as well as the plain blank image ---
            crop_img = gray[Y - 1:Y + H + 1, X - 1:X + W + 1]
            zero_pixel = (np.sum(crop_img == 0) / crop_img.size) * 100
            if zero_pixel >= 12:
                cv2.rectangle(src_img, (X , Y ), (X + W , Y + H ), (0, 0, 255), 2)
                x_vals.append(X - 1)
                y_vals.append(Y - 1)
                w_vals.append(X + W + 1)
                h_vals.append(Y + H + 1)
                all_b.append([X, Y, W, H])

    '''fixed_boxes = combine_boxes(all_b)
    for box in fixed_boxes:
        [X, Y, W, H] = box
        cv2.rectangle(src_img, (X, Y), (X + W, Y + H), (0, 0, 255), 2)
    cv2.imshow("bb", src_img)
    cv2.waitKey()'''
    data = {'X': x_vals, 'Y': y_vals, 'W': w_vals, 'H': h_vals}
    df = pd.DataFrame(data)
    df1 = df.sort_values(by=['Y'])
    bbox = df1.values[:, 0:4]
    y_val = bbox[0][1]
    line_index = []
    line_num = 1
    for k, box in enumerate(bbox):
        y_curr = box[1]
        if abs(y_val - y_curr) <= 25:
            line_index.append(line_num)
        else:
            y_val = y_curr
            line_num += 1
            line_index.append(line_num)
    df1["lineNumber"] = line_index
    df1 = df1.groupby('lineNumber').apply(pd.DataFrame.sort_values, 'X')
    array = df1[['X', 'Y', 'W', 'H']].values
    fixed_boxes = combine_boxes(array)
    fixed_boxes = combine_boxes(fixed_boxes)

    for box in fixed_boxes:
        [X, Y, W, H] = box
        cv2.rectangle(src_img1, (X, Y), (W,  H), (0, 0, 255), 2)
    cv2.imshow("bb", src_img)
    cv2.imshow("bb1", src_img1)
    cv2.waitKey()
    return df1