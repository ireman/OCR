import cv2
import numpy as np
from draw_bb import remove_lines, load_images_from_folder
from removeDiagonalLines import remove_diag_lines, remove_headline

img_list = load_images_from_folder('Cyrus_tiny')
for i, img in enumerate(img_list):
    #img = remove_headline(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # --- perform morphological operation to ensure smaller portions are part of a single character ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 80))
    threshed = cv2.morphologyEx(th2, cv2.MORPH_GRADIENT, kernel)
    # --- find contours ---
    Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in Contours:
        [X, Y, W, H] = cv2.boundingRect(contour)
        if W * H > 100000:
            img = img[Y + 105: Y + H - 80, X + 65: X + W - 20]
            #cv2.rectangle(src, (X + 50, Y + 90), (X - 20 + W, Y - 65 + H), (0, 0, 255), 2)
    remove_headline(img)
    remove_lines(img)
    remove_diag_lines(img)
    #cv2.imshow('result', img)
    #cv2.waitKey()
    cv2.imwrite('cyrus_clean_img\Cyrus_page_' + str(i+35) + '.png', img)