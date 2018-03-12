import cv2, os
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(img_dir, img_file):
    return mpimg.imread(os.path.join(img_dir, img_file.strip()))

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

def adjust_image(img_dir, center, left, right, steering_angle):
    '''adjust the stering angle'''
    choise = np.random.choise(3)
    if choise == 0:
        return load_image(img_dir, left), steering_angle + 0.2
    elif choise == 1:
        return load_image(img_Dir, right), steering_angle - 0.2
    return load_image(img_dir, center), steering_angle

def flip_img(image, steering_angle):
    '''randomly flip the image'''
    if np.random.rand()<0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle



def augument(center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(center, left, right, steering_angle)
    image, steering_angle = flip_img(image, steering_angle)

    return image, steering_angle


def batch_generator(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 66, 200, 3))
    labels = np.zeros((batch_size, 1))

    while True:
        for i in range(batch_size):
            index = np.random.choice(len(features), 1)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels
