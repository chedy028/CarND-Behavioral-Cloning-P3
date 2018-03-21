import sklearn, cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from utility import getimage, combineimages,generator
from sklearn.model_selection import train_test_split

def nvidia_CNN():

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

c_camera, l_camera, r_camera, measurements = getimage('data')
images, measurements = combineimages(c_camera, l_camera, r_camera, measurements, 0.2)
print('Total Images: {}'.format( len(images)))

samples = list(zip(images, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = nvidia_CNN()
model.summary()
#model.compile(loss='mse', optimizer='adam')
#object = model.fit_generator(train_generator, samples_per_epoch= \
                 #len(train_samples), validation_data=validation_generator, \
                 #nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

#model.save('model.h5')
#print(object.history.keys())
#print('Loss')
#print(object.history['loss'])
#print('Validation Loss')
#print(object.history['val_loss'])
