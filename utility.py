import cv2, os
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split
INPUT_SHAPE= 66, 200

def flipping(images, angles):
    aug_images, aug_angles = [], []
    for image, angle in zip(images, angles):
        aug_images.append(image)
        aug_angles.append(angle)
        # Flipping Images (horizontally) and inverting Steering Angles
        aug_images.append(np.fliplr(image))
        aug_angles.append(angle * -1.0)
    return aug_images, aug_angles

def resize(images):
    '''Resize the image to the input shape used by the CNN'''
    return cv2.resize(images, INPUT_SHAPE)
def augment(images, angles):
    flipping(images, angles)
    resize(images)
    return images,angles
