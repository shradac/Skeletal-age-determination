import pandas as pd
import os
import cv2

csv_path = '../Minor Project/input/csv/450images.csv'
path = '../Minor Project/input/images/'
save_path = '../Minor Project/input/resized/'

csv = pd.read_csv(csv_path)
files = os.listdir(path)

height = []
width = []
columns=['id','boneage','male','boneage in years']

for index, row in csv.iterrows():
    filename = path + str(row['id']) + '.png'
    image = cv2.imread(filename)
    height.append(image.shape[0])
    width.append(image.shape[1])
    
    
print("Height = ", sum(height)/len(height))
print("Width =  ", sum(width)/len(width))


for index, row in csv.iterrows():
    filename = path + str(row['id']) + '.png'
    image = cv2.imread(filename)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image_bw,(1600,1300))
    cv2.imwrite(save_path + str(row['id']) + '.png',image)

