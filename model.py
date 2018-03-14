import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Convolution2D, MaxPooling2D, Dropout, Dense, Flatten,Cropping2D
from utility import augment
import csv
import cv2
import numpy as np


'''data_dir = './data'
data_df = pd.read_csv('./data/driving_log.csv')
X = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values'''

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center = './data/IMG/'+batch_sample[0].split('/')[-1]
                left = './data/IMG/'+batch_sample[1].split('/')[-1]
                right = './data/IMG/'+batch_sample[1].split('/')[-1]
                image_center = cv2.imread(center)
                image_left = cv2.imread(left)
                image_right = cv2.imread(right)
                angle = float(batch_sample[3])
                images.append([image_center, image_left, image_right])
                angles.append(angle)


            # trim image to only see section with road

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
'''


# Load the csv file to get the figure name
lines = []
with open ('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

del(lines[0])#delete the header

images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i] #read the middle, left, right images
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename # for running code in AWS
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

X = np.array(images)
y = np.array(measurements)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)'''
def my_resize_function(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (18, 60)



model = Sequential
model.add(Cropping2D(cropping=((70, 25), (1, 1)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: my_resize_function(x)))
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(18,60,3)))
model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mean_squared_error', optimizer='Adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
