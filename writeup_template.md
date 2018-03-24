# **Behavioral Cloning**


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start off a simple neural network and consistently improve the model by going deeper and using data augmentation.

My first step was to use a simple convolutional neural network model. I preprocessed the data by split image and steering angle data into a training and validation with a ratio of 80% and 20% respectively.

I directly follow the nvidia paper and kept simplylifing and optimising it while make sure it performs well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

_________________________________________________________________
|Layer (type)                 |Output Shape       |       Param #   |
|-----------------------------|-------------------|-----------------|
|lambda_1 (Lambda)            |(None, 160, 320, 3)|       0         |
|cropping2d_1 (Cropping2D)    |(None, 90, 320, 3) |       0         |
|conv2d_1 (Conv2D)            |(None, 43, 158, 24)|       1824      |
|conv2d_2 (Conv2D)            |(None, 20, 77, 36) |       21636     |
|conv2d_3 (Conv2D)            |(None, 8, 37, 48)  |       43248     |
|conv2d_4 (Conv2D)            |(None, 6, 35, 64)  |       27712     |
|conv2d_5 (Conv2D)            |(None, 4, 33, 64)  |       36928     |
|flatten_1 (Flatten)          |(None, 8448)       |       0         |
|dense_1 (Dense)              |(None, 100)        |       844900    |
|dense_2 (Dense)              |(None, 50)         |       5050      |
|dense_3 (Dense)              |(None, 10)         |       510       |
|dense_4 (Dense)              |(None, 1)          |       11        |

Total params: 981,819

Trainable params: 981,819

Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process

I used the data provided by udacity this time

![](https://i.imgur.com/oYWRvgQ.jpg)

To augment the data sat, I also used multiple cameras (left and right cameras) and i adjusted the steering angle accordingly. Some images from the left and right cameras can be seen below:

![](https://i.imgur.com/AwHehoB.jpg)
![Imgur](https://i.imgur.com/XLpH10Q.jpg)


The preprocessing is being conducted in the model network with two layers:

- Normalizing image with mean value 0 and boundaries: -0.5 to 0.5
- Cropping image (70 pixels from top and 25 from bottom)
After the collection process, I had 24108 number of data points. I then preprocessed this data by creating 24108 training samples and 4822 validation samples.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by validation and training loss diagram. The final solution diagram can be seen below:

![](http://i.imgur.com/lhaRuTj.jpg)

 I used an adam optimizer so that manually training the learning rate wasn't necessary.

The optimal batch size is 32 since any larger value results in more epochs and different model architecture.

Please see the mp4 file for the final video. 
