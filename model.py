import csv
import cv2
import numpy as np

# Load the csv file to get the figure name
lines = []
with open ('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

del(lines[0])#delete the header

image = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i] #read the middle, left, right images
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename # for running code in AWS
        image = cv2.imread(current_path)
        measurement = float(line[3])
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
modeladd(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.save('model.h5')
