#** Behavioral Cloning** 


** Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Model
The Model that teach the car to drive autonomously is a slightly modified version of LeNet. We will discuss the model architecture and training strategy in the later section.

## Dataset 
The data used to train the autonomous car in the simulator is a set of images with corresponding steering angles. A steering angle is a float number with a range from -1 to 1, where the negative value corresponds to a left turn and positive value correspond to a right turn. A single data point contains three images: left, center, and right taken by the three different cameras on the car. I assign the angle to the central image and assign an offset to the left and right images. This offset helps the car to recover to the center if it goes off the course.

## Data Augmentation
In order to mitigate the issue that the autonomous car pulls too hard to the left, I augment the data to expand the training data set to help model generalize better. One of the approaches I used is to flip the images horizontally, then inverted the steering angles accordingly. Other data augmentation technique that we can potentially use are changing the brightness of the images, flipping it vertically or adding random shadows to the images.

## Data Preprocessing
### Normalization
Each channel in the image is normalized by the maximum possible value (225) to a range between 0 to 1, and mean center the image by subtracting 0.5 to zero. 

```
lambda x: x / 255.0 - 0.5
```

### Crop 
Not all the information in the images are useful when training the model. In the training set, I observed that the top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. These portions are distracting the model to predict the steering angle. I crop each image to focus on only the portion of the image that is useful to the model.


## Recovery Data

Another interesting observation I have in the training set is that a great portion of the data provided by Udacity is captured when the car is driving straight. The model trained with the data is particularly good at driving on the straight course and terribly bad at making big turns. To mitigate the problem, I collect more data specifically on the car recovering from unfavorable positions on the track, i.e. when the car is about to go off the course, and making big turns. I also randomly removed the data with steering angle less than `0.1` making training set more evenly distributed with driving straight and making turns to help model generalize better.

## Model Architecture and Training Strategy
### Data, lots of data
From the start of the project, my strategy is not to implement a fancy Deep Neural Network (DNN) with tons of parameters to tune but instead a simple DNN model with lots of good quality  data. After the collection and data augmentation process, I had around 50k number of data points.

### LeNet
I used a slightly modified version of Lenet as my model; the following is the detail of the model for each layer in DNN.


```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_61 (Lambda)               (None, 160, 320, 3)   0           lambda_input_61[0][0]            
____________________________________________________________________________________________________
cropping2d_56 (Cropping2D)       (None, 60, 320, 3)    0           lambda_61[0][0]                  
____________________________________________________________________________________________________
convolution2d_151 (Convolution2D (None, 56, 316, 6)    456         cropping2d_56[0][0]              
____________________________________________________________________________________________________
maxpooling2d_92 (MaxPooling2D)   (None, 28, 158, 6)    0           convolution2d_151[0][0]          
____________________________________________________________________________________________________
convolution2d_152 (Convolution2D (None, 24, 154, 16)   2416        maxpooling2d_92[0][0]            
____________________________________________________________________________________________________
maxpooling2d_93 (MaxPooling2D)   (None, 12, 77, 16)    0           convolution2d_152[0][0]          
____________________________________________________________________________________________________
flatten_53 (Flatten)             (None, 14784)         0           maxpooling2d_93[0][0]            
____________________________________________________________________________________________________
dense_158 (Dense)                (None, 120)           1774200     flatten_53[0][0]                 
____________________________________________________________________________________________________
activation_105 (Activation)      (None, 120)           0           dense_158[0][0]                  
____________________________________________________________________________________________________
dense_159 (Dense)                (None, 84)            10164       activation_105[0][0]             
____________________________________________________________________________________________________
activation_106 (Activation)      (None, 84)            0           dense_159[0][0]                  
____________________________________________________________________________________________________
dense_160 (Dense)                (None, 1)             85          activation_106[0][0]             
====================================================================================================
Total params: 1,787,321
Trainable params: 1,787,321
Non-trainable params: 0

```


* The model includes RELU layers to introduce nonlinearity
* The model contains max pooling layers in order to reduce overfitting. 
* The model used an adam optimizer, so the learning rate was not tuned manually
* The image and steering angle data is split into a training and validation set with ratio of 80/20

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

![alt text](https://github.com/danny2000tw/CarND-Behavioral-Cloning-P3/blob/master/ezgif.com-gif-maker.gif "Logo Title Text 1")


## Future improvements
* Try better DNN networks, i.e. [Nvidia DNN](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
* Data Augmentation, i.e. add random brightness, saturation and shadows into training set.
