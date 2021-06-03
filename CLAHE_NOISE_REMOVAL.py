import cv2
import os
import numpy as np


path = '../Minor Project/input/resized/'
save_path = '../Minor Project/input/no-bg clahe/'


for filename in os.listdir(path):
    img = cv2.imread(path+filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)
    for i in img_contours:
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [i],-1, 255, -1)
        new_img = cv2.bitwise_and(img, img, mask=mask)
        
    clahe = cv2.createCLAHE(clipLimit = 2)
    final_img = clahe.apply(new_img) + 30
    cv2.imwrite(os.path.join(save_path , filename),final_img)
