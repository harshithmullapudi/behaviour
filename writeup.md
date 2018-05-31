#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/1.png
[image2]: ./examples/2.jpg
[image3]: ./examples/3.jpg
[image4]: ./examples/4.jpg


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes
1. model.py - Which the Neural neteork model used to train with the respective preprocessings.
2. Writeup.md - Which contains details of the project.
3. drive.py - Which helps in have a real usage of the prepared model
4. video.py - which is used to make a video of the images.

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.py has all code which is used to the train the model with the help of the images from the simulator and also respective code for the validation and the code is also modulated for easy understanding and readablity.


###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of the Convolution neural network with depths of 32 to 128 between lines 108 - 124

The model has the relu layers, Conv2d layers with the respective activations and droupouts which you can see in the code.

####2. Attempts to reduce overfitting in the model

I preferred droupouts in case to reduce the overfitting of the data.

The model was trained on different datsets to ensure it doesn't overfit and it works perfectly in the problem. I have also checked this with the simulator.

####3. Model parameter tuning

The model used adam optimizer and the other parameters such as learning rate, epochs have tuned manually with changing multiple times by checking the perfect suiting
model.

####4. Appropriate training data

I have initially used the udacity training dataset later I have created my own dataset and started to train on both of the datasets to see that it won't overfit and parallely it learns and gives the best model. So I have trained on multiple datasets to ensure everything is going fine.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The Strategy was.

1. initially I collected the dataset which will be the basic step in order to train the model which I have done using all cameras both center and sides.

2. Later I have designed a small model and trained then checked with the simulator which went very bad and thus took help of my classes in udacity and choosed udacity suggested model. 

3. Then wrote the code for the model to train and used multiple preprocessing steps such as cropping, shadow, brightness, augmentation etc.

4. Finally trained my model on the respective datasets and at the end the car started to go automatically.

####2. Final Model Architecture

The final model used was mentioned in the code and that was tested and works fine with the respective simulator also.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

1. First to train I directly used the udacity training dataset.

2. Later to get good driving skills I have taken all images from the center camera.

3. Later for the model to perfectly go around the curves and to understand to move in center I have taken images from the sides.


Here are photos of the traning data collected.

![alt text][image2]
![alt text][image3]
![alt text][image4]


