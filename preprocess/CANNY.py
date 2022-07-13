import cv2
import pytesseract
import unidecode
import numpy as np

def remove_diag_line(image):
    #image = cv2.imread('pic.PNG')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
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
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    return cv2.merge([image, image, image])


words_to_remove = ['1', '2', '3', 'a']
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread("Cyrus\Cyrus-page-035.jpg")
#image = remove_diag_line(image)

'''text = pytesseract.image_to_string(image)
print(unidecode.unidecode(text))'''

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
inverted_thresh = 255 - thresh
dilate = cv2.dilate(inverted_thresh, kernel, iterations=2)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    '''ROI = thresh[y:y+h, x:x+w]
    data = pytesseract.image_to_string(ROI, lang='eng',config='--psm 6').lower()
    print(unidecode.unidecode(data))
    if data.split() in words_to_remove:
        image[y:y+h, x:x+w] = [255,255,255]'''

    [X, Y, W, H] = cv2.boundingRect(c)
    if 100 < cv2.contourArea(c) < 8000:
    #--- draw those bounding boxes in the actual image as well as the plain blank image ---
        cv2.rectangle(image, (X, Y), (X + W, Y + H), (0,0,255), 2)
dilate = cv2.merge([dilate, dilate, dilate])
result = cv2.bitwise_xor(dilate, image)
#cv2.imwrite("OCR_result.PNG", 255-result)
#cv2.imshow("thresh", thresh)
imSdilate = cv2.resize(dilate, (960, 740))
imSresult = cv2.resize(result, (960, 740))

cv2.imshow("dilate", imSdilate)
cv2.imshow("image", imSresult)
#cv2.imshow("result",255-result)
cv2.waitKey(0)
'''data = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6').lower()
print(data)
text = pytesseract.image_to_string(255-result)
print(unidecode.unidecode(text))'''