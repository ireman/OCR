from pdf2image import convert_from_path
import cv2
import numpy as np
from draw_bb import remove_lines, load_images_from_folder
from removeDiagonalLines import remove_diag_lines, remove_headline
from PIL import Image

ROI_img_list = []
Y_offset = 0

images = convert_from_path('Cyrus.pdf', poppler_path=r"poppler-21.03.0\Library\bin")[34:]
#images = load_images_from_folder('Cyrus')
headline_pos_list = []
for i in range(len(images)):
    src = np.array(images[i])
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    zero_line = np.zeros(20)
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    horizontal_kernel = np.array([[1, 1, 1],
                                  [0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0],
                                  [1, 1, 1]], dtype=np.uint8)
    horizontal_kernel = np.repeat(horizontal_kernel, repeats=20, axis=1)
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])

    prev_y_pos = 0
    for c in sorted_ctrs:
        [X, Y, _, _] = cv2.boundingRect(c)
        if Y > prev_y_pos + 40:
            headline_pos_list.append(Y + Y_offset)
        prev_y_pos = Y
    ROI_img_list.append(Image.fromarray(src))
    Y_offset += thresh.shape[0]
min_shape = sorted([(np.sum(i.size), i.size) for i in ROI_img_list])[0][1]
imgs_comb = 255-np.vstack((np.asarray(img.resize((min_shape[0],img.size[1]))) for img in ROI_img_list))

for i,pos in enumerate(headline_pos_list):
    if i < len(headline_pos_list)-1:
        Y1 = headline_pos_list[i]
        Y2 = headline_pos_list[i+1]
        cv2.imwrite('separateCyrusImg\\'+str(i)+'.png', 255-imgs_comb[Y1-30: Y2-30, :])
    else:
        cv2.imwrite('separateCyrusImg\\'+str(i)+'.png', 255-imgs_comb[Y2-30:, :])