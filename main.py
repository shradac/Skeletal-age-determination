import numpy as np 
import pandas as pd 
import tensorflow as tf
import datetime, os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

train_df = pd.read_csv('../Minor Project/input/csv/aug.csv')

train_df['id'] = train_df['id'].apply(lambda x: str(x)+'.png')

train_df.head()

train_df[['boneage']].hist(figsize=(5,5))

train_df['gender'] = train_df['male'].apply(lambda x: 'male' if x else 'female')
print(train_df['gender'].value_counts())
sns.countplot(x = train_df['gender'])


print('MAX age: ' + str(train_df['boneage'].max()) + ' months')

print('MIN age: ' + str(train_df['boneage'].min()) + ' months')


mean_bone_age = train_df['boneage'].mean()
print('mean: ' + str(mean_bone_age))


print('median: ' +str(train_df['boneage'].median()))


std_bone_age = train_df['boneage'].std()


train_df['bone_age_z'] = (train_df['boneage'] - mean_bone_age)/(std_bone_age)

print(train_df.head())

male = train_df[train_df['gender'] == 'male']
female = train_df[train_df['gender'] == 'female']

df_train, df_valid = train_test_split(train_df, test_size = 0.15, random_state = 0)


img_h = 1600
img_w = 1300

train_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
val_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_data_generator.flow_from_dataframe(
    dataframe = df_train, directory = '../Minor Project/input/no-bg clahe',
    x_col= 'id', y_col= 'boneage', batch_size = 32, seed = 42,
    shuffle = True, class_mode= 'raw',
    color_mode = 'grayscale', target_size = (img_h, img_w))

validation_generator = val_data_generator.flow_from_dataframe(
    dataframe = df_valid, directory = '../../Minor Project/input/no-bg clahe',
    x_col= 'id', y_col= 'boneage', batch_size = 32, seed = 42,
    shuffle = True, class_mode= 'raw',
    color_mode = 'grayscale', target_size = (img_h, img_w))

test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

test_generator = test_data_generator.flow_from_directory(
    directory = '../Minor Project/input/no-bg clahe',
    shuffle = False, 
    class_mode = None,
    color_mode = 'grayscale',
    target_size = (img_h,img_w))

test_X, test_Y = next(val_data_generator.flow_from_dataframe( 
    dataframe = df_valid, directory = '../Minor Project/input/no-bg clahe',
    x_col = 'id', y_col = 'boneage', 
    target_size = (img_h, img_w),
    batch_size = 907,
    color_mode = 'grayscale',
    class_mode = 'raw'
    ))



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, AveragePooling2D
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.losses import Huber

model = Sequential()
model.add(Conv2D(64, kernel_size=(7,7), strides=(5,5), padding='same', activation='relu', input_shape=(img_h, img_w, 1)))
model.add(Conv2D(64, kernel_size=(7,7), strides=(5,5), padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=(3,3), strides=(3,3), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(5,5), strides=(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(128, kernel_size=(5,5), strides=(3,3), padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(256, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(256, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(512, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(512, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(1024, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(1024, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))) #CHECK FOR 512
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='relu'))

model.compile(loss=Huber(), optimizer='adam', metrics=['mape','mse','mae'])

model.summary()


from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('bone_age')

checkpoint = ModelCheckpoint(weight_path, monitor='val_mae', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_mae', factor=0.8, patience=5, verbose=1, mode='auto', epsilon=0.001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_mae", mode="min", patience=15)
callbacks_list = [checkpoint, early, reduceLROnPlat]

history = model.fit_generator(train_generator, 
    validation_data = (test_X, test_Y), 
    epochs = 100, 
    callbacks = callbacks_list)

model.load_weights(weight_path)


model.save('custom.h5')



from tensorflow import keras
model = keras.models.load_model('custom.h5')
pred = model.predict(test_X, batch_size = 32, verbose = 2)
results = model.evaluate(test_X, test_Y)
print(results)


fig, axss = plt.subplots(figsize = (7,7))
axss.plot(history.history['loss'])
axss.plot(history.history['val_loss'])
axss.legend(['train', 'test'], loc='upper left')
axss.set_xlabel('epoch')
axss.set_ylabel('loss')
fig.savefig('loss.png', dpi = 300)


fig, axs = plt.subplots(figsize = (7,7))
axs.plot(history.history['mape'])
axs.plot(history.history['val_mape'])
axs.legend(['train', 'test'], loc='upper left')
axs.set_xlabel('epoch')
axs.set_ylabel('mape')
fig.savefig('mape.png', dpi = 300)


fig, axc = plt.subplots(figsize = (7,7))
axc.plot(history.history['mae'])
axc.plot(history.history['val_mae'])
axc.legend(['train', 'test'], loc='upper left')
axc.set_xlabel('epoch')
axc.set_ylabel('mae')
fig.savefig('mae.png', dpi = 300)


fig, axx = plt.subplots(figsize = (7,7))
axx.plot(history.history['mse'])
axx.plot(history.history['val_mse'])
axx.legend(['train', 'test'], loc='upper left')
axx.set_xlabel('epoch')
axx.set_ylabel('mse')
fig.savefig('mse.png', dpi = 300)

df = pd.DataFrame(columns = ['actual','predicted'])
for i in range(len(pred)):
    df.loc[len(df.index)]=[test_Y[i],pred[i][0]]
    
print(df)
save_path = '../Minor Project/'
df.to_csv(save_path + 'actual vs pred.csv')


test_months = (test_Y)

ord_ind = np.argsort(test_Y)
ord_ind = ord_ind[np.linspace(0, len(ord_ind)-1, 12).astype(int)] 
fig, axs = plt.subplots(6, 2, figsize = (15, 30))
for (ind, ax) in zip(ord_ind, axs.flatten()):
    ax.imshow(test_X[ind, :,:,0], cmap = 'bone')
    ax.set_title('Age: %fY\nPredicted Age: %fY' % (test_months[ind]/12.0, pred[ind]/12.0))
    ax.axis('off')
fig.savefig('trained_image_predictions.png', dpi = 300)

fig, ax = plt.subplots(figsize = (7,7))
ax.plot(test_months, pred, 'r.', label = 'predictions')
ax.plot(test_months, test_months, 'b-', label = 'actual')
ax.legend(loc = 'upper right')
ax.set_xlabel('Actual Age (Months)')
ax.set_ylabel('Predicted Age (Months)')
fig.savefig('pred.png', dpi = 300)
