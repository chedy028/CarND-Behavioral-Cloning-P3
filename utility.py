import cv2, os
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3

def crop(image):
    '''removing the sky at the top and the car front at the bottom'''
    return image[60:-25,:,:]

def resize(image):
    '''Resize the image to the input shape used by the CNN'''
    return cv2.resize(image, )

def rgb2yuv(image):
    '''
    Convert the imgage from RBG to YUV
    '''
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    '''combine all function above'''
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image
