import cv2
import csv
import numpy as np
import os
import sklearn

def getdata(path):

    lines = []
    with open(path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            lines.append(line)
    del (lines[0]) #delete the headers
    return lines


def getimage(path):

    directories = [x[0] for x in os.walk(path)]
    data = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centerCamera = []
    leftCamera = []
    rightCamera = []
    measurement = []
    for directory in data:
        lines = getdata(directory)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        centerCamera.extend(center)
        leftCamera.extend(left)
        rightCamera.extend(right)
        measurement.extend(measurements)

    return centerCamera, leftCamera, rightCamera, measurement

def combineimages(center, left, right, measurement, correction):

    image = []
    image.extend(center)
    image.extend(left)
    image.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (image, measurements)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)
