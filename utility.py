import cv2
import csv
import numpy as np
import os
import sklearn

def getdata(path):
    """
    get list of data from driving_log csv
    """
    lines = []
    with open(path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            lines.append(line)
    del (lines[0]) #delete the headers
    return lines


def getimage(path):
    """
    Finds all the images needed for training on the path `dataPath`.
    Returns `([centerPaths], [leftPath], [rightPath], [measurement])`
    """
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

def combineImages(center, left, right, measurement, correction):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([imagePaths], [measurements])
    """
    image = []
    image.extend(center)
    image.extend(left)
    image.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (image, measurements)
