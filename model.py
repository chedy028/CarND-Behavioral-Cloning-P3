import pandas as pd
import csv
import cv2
import numpy as np
import utility
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


X_raw = np.array(images)
y_raw = np.array(measurements)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_raw, y_raw, test_size = 0.2, random_state=0)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(24,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.save('model.h5')
