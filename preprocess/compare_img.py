import os
import numpy as np
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder,filename)
        for img in os.listdir(path):
            stream = open(os.path.join(path,img), "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_GRAYSCALE)
            #img = cv2.imread(os.path.join(path,img))
            if bgrImage is not None:
                images.append(bgrImage)
    return images

from collections import OrderedDict

'''img_list1 = load_images_from_folder(r"C:\\Users\Home\Downloads\consensus_data11\𒂼")
img_list2 = load_images_from_folder(r"C:\\Users\Home\Downloads\consensus_data11\𒂼")
copy1 = img_list1.copy()
copy2 = img_list2.copy()
k = 0
for i, img1 in enumerate(img_list1):
    for j, img2 in enumerate(img_list2):
        if img1.shape == img2.shape:
            if np.allclose(img1, img2):
               k += 1

a = 5'''