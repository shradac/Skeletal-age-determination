
import numpy as np
from numpy import expand_dims
from matplotlib import pyplot
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import pandas as pd

dir_path = '../Minor Project/input/resized/'
save_path = '../Minor Project/input/aug/'
csv_path = '../Minor Project/input/csv/450images.csv'

csv = pd.read_csv(csv_path)
files4 = []
age4 = []

files16 = []
age16 = []

files17 = []
age17 = []

for index, row in csv.iterrows():
    if row['boneage in years'] == 4:
        files4.append(str(row['id']) + '.png')
        age4.append(str(row['boneage']))
    if row['boneage in years'] == 16:
        files16.append(str(row['id']) + '.png')
        age16.append(str(row['boneage']))
    if row['boneage in years'] == 17:
        files17.append(str(row['id']) + '.png')
        age17.append(str(row['boneage']))

data_gen = ImageDataGenerator(zoom_range=0.2,brightness_range=(0.7,1.1))

count=0
for filename in files4:    
    image_path = os.path.join(dir_path, filename)
    image = np.expand_dims(img_to_array(load_img(image_path)), 0)
    data_gen.fit(image)
    for x, val in zip(data_gen.flow(image, save_to_dir=save_path + '4/', save_prefix=age4[count],save_format='png'),range(1)):
        pass
    count+=1

print(count)

count=0
for filename in files16:    
    image_path = os.path.join(dir_path, filename)
    image = np.expand_dims(img_to_array(load_img(image_path)), 0)
    data_gen.fit(image)
    for x, val in zip(data_gen.flow(image, save_to_dir=save_path + '16/', save_prefix=age16[count],save_format='png'),range(1)):
        pass
    count+=1

print(count)

count=0
for filename in files17:    
    image_path = os.path.join(dir_path, filename)
    image = np.expand_dims(img_to_array(load_img(image_path)), 0)
    data_gen.fit(image)
    for x, val in zip(data_gen.flow(image, save_to_dir=save_path + '17/', save_prefix=age17[count],save_format='png'),range(1)):
        pass
    count+=1

print(count)


i = 0
for filename in os.listdir(save_path + '4/'):
    csv = csv.append({'id':'a4'+ str(i).zfill(4),'boneage':filename.partition('_')[0], 'male':'aug','boneage in years':'4' },ignore_index=True)
    os.rename(save_path + '4/' + filename,save_path + '4/' + 'a4' + str(i).zfill(4) + '.png')
    i+=1
    
i = 0
for filename in os.listdir(save_path + '16/'):
    csv = csv.append({'id':'a16'+ str(i).zfill(4),'boneage':filename.partition('_')[0], 'male':'aug','boneage in years':'16' },ignore_index=True)
    os.rename(save_path + '16/' + filename,save_path + '16/' + 'a16' + str(i).zfill(4) + '.png')
    i+=1

i = 0
for filename in os.listdir(save_path + '17/'):
    csv = csv.append({'id':'a17'+ str(i).zfill(4),'boneage':filename.partition('_')[0], 'male':'aug','boneage in years':'17' },ignore_index=True)
    os.rename(save_path + '17/' + filename,save_path + '17/' + 'a17' + str(i).zfill(4) + '.png')
    i+=1
    
    
csv.to_csv('../Minor Project/input/csv/aug.csv')
