import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms


def bbox2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    rows = np.any(thresh, axis=1)
    cols = np.any(thresh, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax + 1, xmin:xmax + 1]


def union(box1, box2):
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = max(box1[2], box2[2])
    h = max(box1[3], box2[3])
    return x, y, w, h


def intersection(box1, box2):
    if 2 < box1[2] - box2[0] < 50 and abs(box1[3] - box2[3]) < 20:
        return True
    return False


def combine_boxes(boxes):
    new_array = []
    i = 0
    while i < len(boxes) - 1:
        if intersection(boxes[i], boxes[i + 1]):
            new_array.append(union(boxes[i], boxes[i + 1]))
            i += 2
        else:
            new_array.append(boxes[i])
            i += 1
    if i <= len(boxes) - 1:
        new_array.append(boxes[i])
    return np.array(new_array).astype('int')


class AnnotationAndPrediction:
    """
    This class provide a full solution for the annotation of cuneiform images
    The class contains two main methods:
    1. Automatically detects the location of the various cuneiform signs and the line in which they are located.
    2. Unicode prediction appropriate for each sign.
    """

    def __init__(self,
                 model_path='cyrus_classifier_filtered_test_not_aug.pt',
                 labels_csv='filtered_label.csv',
                 unicode_to_label_dict='unicode_to_label_dict.csv',
                 label_to_unicode_dict='label_to_unicode_dict.csv'):

        df = pd.read_csv(labels_csv, keep_default_na=False, encoding='utf-8')
        self.label_list = df['label'].tolist()

        df = pd.read_csv(unicode_to_label_dict, keep_default_na=False, encoding='utf-8')
        self.unicode_to_label_dict = df.set_index('label').T.to_dict('list')
        self.unicode_to_label_dict = {k: list(filter(None, v)) for k, v in self.unicode_to_label_dict.items()}

        df = pd.read_csv(label_to_unicode_dict, keep_default_na=False, encoding='utf-8', header=None)
        self.label_to_unicode_dict = df.set_index(0).astype(str).agg(' '.join, axis=1).to_dict()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(self.label_list))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to(self.device)
        model = model.double()
        self.model = model

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
        for contour in Contours:
            [X, Y, W, H] = cv2.boundingRect(contour)

            area = cv2.contourArea(contour)
            if 8 < W < 130 and 8 < H < 50 and area > 60:
                # --- draw those bounding boxes in the actual image as well as the plain blank image ---
                crop_img = gray[Y - 1:Y + H + 1, X - 1:X + W + 1]
                zero_pixel = (np.sum(crop_img == 0) / crop_img.size) * 100
                if zero_pixel >= 12:
                    x_vals.append(X)
                    y_vals.append(Y)
                    w_vals.append(X + W)
                    h_vals.append(Y + H)

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
        df1 = pd.DataFrame()
        df1['X'] = fixed_boxes[:, 0]
        df1['Y'] = fixed_boxes[:, 1]
        df1['W'] = fixed_boxes[:, 2] - fixed_boxes[:, 0]
        df1['H'] = fixed_boxes[:, 3] - fixed_boxes[:, 1]

        bbox = df1.values[:, 0:4]
        y_val = bbox[0][1]
        line_index = []
        line_num = 1
        for k, box in enumerate(bbox):
            y_curr = box[1]
            if abs(y_val - y_curr) <= 22:
                line_index.append(line_num)
            else:
                y_val = y_curr
                line_num += 1
                line_index.append(line_num)
        df1["lineNumber"] = line_index
        df1 = df1.groupby('lineNumber').apply(pd.DataFrame.sort_values, 'X')
        return df1

    def get_image_prediction(self, image: np.ndarray, bb_csv: pd.DataFrame) -> pd.DataFrame:
        """
        This function gets an image of cuneiform hand-copy and dataFrame with bounding box
        and return adding column contain prediction Unicode for each bounding box.
        """
        df = bb_csv
        bbox = df.values[:, 0:5]
        predictions_list = []
        txt = ''
        line_num = 1
        m = 0
        crop_img_list = []
        for box in bbox:
            X, Y, W, H = box[0:4]
            curr_line = box[4]
            crop_img1 = image[Y:Y + H, X:X + W]
            crop_img = cv2.resize(crop_img1, dsize=(64, 64))
            crop_img = (np.asarray(crop_img).reshape(64, 64, 3).astype(np.float64)) / 255
            trans = transforms.ToTensor()
            crop_img = trans(crop_img)
            crop_img = crop_img.to(device=self.device)
            crop_img_list.append(crop_img)
        torch_images = torch.stack(crop_img_list)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(torch_images)
            for score in scores:
                _, predictions = score.max(0)
                predictions_list.append(self.label_list[predictions])
        df['predictions'] = predictions_list

        return df

    def get_detexify_prediction(self, image: np.ndarray) -> list:
        """
        This function gets an image of one cuneiform hand-copy
        and return list of tuples with 5 topk prediction and confidence.
        """
        crop_img = bbox2(image)
        crop_img = cv2.resize(crop_img, dsize=(64, 64))
        crop_img = (np.asarray(crop_img).reshape(64, 64, 3).astype(np.float64)) / 255

        trans = transforms.ToTensor()
        crop_img = trans(crop_img)
        crop_img = crop_img.to(device=self.device)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(crop_img[None, ...])
            probs = torch.nn.functional.softmax(scores, dim=1)
            conf, predict = torch.topk(probs, 5)
            pred_list = predict.flatten().tolist()
            pred_label = [self.label_list[pred] for pred in pred_list]
            conf_list = conf.flatten().tolist()
            pred_with_conf_list = list(zip(pred_label, conf_list))
        return pred_with_conf_list
