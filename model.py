import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utility import batch_generator
import csv
import cv2
import numpy as np
'''
data_dir = './data'
data_df = pd.read_csv('./data/driving_log.csv')

X = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)'''


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

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

model = Sequential()
model.add(Conv2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error')
model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.004))

generator = batch_generator(X_train, y_train, 10)
model.fit_generator(generator, samples_per_epoch=50, nb_epoch=10)
