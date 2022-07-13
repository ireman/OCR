from pdf2image import convert_from_path
import cv2
import numpy as np
from draw_bb import remove_lines
from removeDiagonalLines import remove_diag_lines, remove_headline
from PIL import Image

images = convert_from_path('Cyrus.pdf', poppler_path=r"poppler-21.03.0\Library\bin")[34:]
headline_pos_list =[]
clean_images_list = []
ROI_img_list = []
Y_offset = 0
for i in range(len(images)):
    src = np.array(images[i])
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal

    # --- perform morphological operation to ensure smaller portions are part of a single character ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 80))
    threshed = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    # --- find contours ---
    Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in Contours:
        [X, Y, W, H] = cv2.boundingRect(contour)
        if W * H > 100000:
            src = src[Y + 105: Y + H - 80, X + 65: X + W - 20]
            #cv2.rectangle(src, (X + 50, Y + 90), (X - 20 + W, Y - 65 + H), (0, 0, 255), 2)
    ROI_img_list.append(Image.fromarray(src))

    zero_line = np.zeros(20)
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    horizontal_kernel = np.array([[1, 1, 1],
                                  [0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0],
                                  [1, 1, 1]], dtype=np.uint8)
    horizontal_kernel = np.repeat(horizontal_kernel, repeats=15, axis=1)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])

    prev_y_pos = 0
    for c in sorted_ctrs:
        [X, Y, W, H] = cv2.boundingRect(c)
        if (Y > prev_y_pos + 40) and (4 < Y < thresh.shape[0]-4):
            headline_pos_list.append(Y + Y_offset)
            prev_y_pos = Y
    Y_offset += thresh.shape[0]

#min_shape = sorted( [(np.sum(i.size), i.size ) for i in ROI_img_list])[0][1]
#min_img_width = min(img.shape[1] for img in ROI_img_list)
min_shape = sorted([(np.sum(i.size), i.size) for i in ROI_img_list])[0][1]
imgs_comb = 255-np.vstack((np.asarray(img.resize((min_shape[0],img.size[1]))) for img in ROI_img_list))
imgs_comb = remove_lines(255-imgs_comb)
imgs_comb = remove_diag_lines(imgs_comb)
#imgs_comb1 = Image.fromarray(imgs_comb)
#imgs_comb1.save('Trifecta_vertical1.png')
'''
find headline between pages
'''
#gray = cv2.cvtColor(imgs_comb, cv2.COLOR_BGR2GRAY)

'''Y,X = src.shape
if len(sorted_ctrs) == 0:
    headline_pos_list.append((i, [X/2, Y]))'''
#remove_headline(src)
#imgs_comb_3D= cv2.merge([imgs_comb, imgs_comb, imgs_comb])
#remove_lines(imgs_comb)
#remove_diag_lines(imgs_comb)

for i,pos in enumerate(headline_pos_list):
    if i < len(headline_pos_list)-1:
        Y1 = headline_pos_list[i]
        Y2 = headline_pos_list[i+1]
        cv2.imwrite('separateCyrusImg\\'+str(i+1)+'.png', imgs_comb[Y1+5: Y2-30, :])
    else:
        cv2.imwrite('separateCyrusImg\\'+str(i+1)+'.png', imgs_comb[Y2+5:, :])
'''clean_images_list.append(src)
#cv2.imshow('result', img)
#cv2.waitKey()
#cv2.imwrite('cyrus_clean_img_1\Cyrus_page_' + str(i+35) + '.png', src)
max_img_width = max(img.shape[1] for img in clean_images_list)
resize_img_list = []
for img in clean_images_list:
# If the image is larger than the minimum width, resize it
height = img.shape[0]
width = img.shape[1]
if width < max_img_width:
    img = img.resize((max_img_width, int(height / width * max_img_width)), Image.ANTIALIAS)
resize_img_list.append(img)
for i in range(len(headline_pos_list)):
if headline_pos_list[i][0] == headline_pos_list[i+1][0]:
    text = resize_img_list[i][
else:
    np.concatenate()'''

