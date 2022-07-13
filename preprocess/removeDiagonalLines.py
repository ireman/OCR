import cv2
import numpy as np

# Read in image, grayscale, and Otsu's threshold
def remove_diag_lines(image):
    #image = cv2.imread('Cyrus_tiny\Cyrus_page_35.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    a = np.array([1,1,1])

    kernel = np.array(np.flip(np.diag(a),1),dtype=np.uint8)
    kernel = np.repeat(kernel, repeats=3, axis=0)

    # Create diagonal kernel
    '''kernel = np.array([[1, 1, 1]
                       [0, 0, 0],
                       [1, 1, 1]], dtype=np.uint8)'''
    opening = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel, iterations=1)
    # Find contours and filter using contour area to remove noise
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1:
            cv2.drawContours(opening, [c], -1, (0,0,0), -1)

    # Bitwise-xor with original image
    #opening = cv2.merge([opening, opening, opening])
    #opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, a, iterations=1)

    result = 255 - cv2.bitwise_xor(thresh, opening)
    return result
    #result=255-thresh-result
    '''cv2.imshow('thresh', thresh)
    cv2.imshow('opening', opening)
    cv2.imshow('result', 255-result)
    #cv2.imwrite("result4.jpg", 255-result)
    
    cv2.waitKey()'''

def remove_headline(image):
    #image = cv2.imread('Cyrus_tiny\Cyrus_page_35.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    for c in cnts:
        [X, Y, W, H] = cv2.boundingRect(c)
        cv2.rectangle(image, (X - 550, Y - 30), (X + 750, Y), (255, 255, 255), -1)
    return image


'''image = cv2.imread('cyrus_clean_img\Cyrus_page_122.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#edges = cv2.Canny(thresh,50,100)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
th2 = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    [X, Y, W, H] = cv2.boundingRect(c)
    if 10 < W < 100 and 10 < H < 50 and cv2.contourArea(c) > 50:
        cv2.rectangle(image, (X, Y), (X + W, Y + H), (0, 0, 255), 2)
        #cv2.drawContours(image, [c], -1, (0,0,255), 2)
cv2.imshow('edges', image)
cv2.waitKey()

a = np.array([1,1,1])

kernel = np.array(np.flip(np.diag(a),1),dtype=np.uint8)
kernel = np.repeat(kernel, repeats=3, axis=0)

# Create diagonal kernel
''''''kernel = np.array([[1, 1, 1]
                   [0, 0, 0],
                   [1, 1, 1]], dtype=np.uint8)''''''
opening = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel, iterations=1)
kernel = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=np.uint8)
closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations=1)
# Find contours and filter using contour area to remove noise
cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area > 1:
        cv2.drawContours(closing, [c], -1, (0,0,0), -1)

# Bitwise-xor with original image
#opening = cv2.merge([opening, opening, opening])
#opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, a, iterations=1)

result = 255 - cv2.bitwise_xor(thresh, closing)
cv2.imshow('result', result)
cv2.imshow('opening', closing)
# cv2.imwrite("result4.jpg", 255-result)
cv2.waitKey()'''