# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/output_10_0.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/00020.png "Traffic Sign 1"
[image5]: ./examples/00028.png "Traffic Sign 2"
[image6]: ./examples/00031.png "Traffic Sign 3"
[image7]: ./examples/00038.png "Traffic Sign 4"
[image8]: ./examples/00043.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  
1. Define the path for training and testing data, and use the pickle.load() function to obtain the features and labels of the training and testing data
2. First get the number of train data, test data and classes, and then plot the images and histogram between each class and correspond number
3. Pre-process the dataset before training, including grayscale and normalization, and then construct the lenet model. Train and test the model in the end.
4. Prepare 20 new dataset on German Traffic Sign Dataset, and make predictions based on trained model.
5. List top 5 probabilities for 20 new images
6. The final testing accuracy is 0.950

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/FrankCAN/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 0
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute on each class

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because in case with traffic signs, the color is unlikely to give any performance boost. But for natural images, color gives extra information, and transforming images to grayscale may hurt performance.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because normalized data has mean zero and equal variance.

I decided to generate additional data because extra data is invariance, and also check if the model is overfitting or underfitting

To add more data to the the data set, some techniques can be used, such as flipping image, or gaussian blur, because some noise and transformation can increase data and also make model robust.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following that augmented data has mean zero and equal variance, and can check the model is overfitting or underfitting


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x48 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x96 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x96 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 3x3x172 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 3x3x172 				|
| Fully connected		| outputs 688.        									|
| Fully connected		| outputs 84.        									|
| Softmax				| outputs 43.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adam optimizer with learning rate 0.001 to minimize the cross entropy, and the epoch is set to 35 and batch size is 128

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1
* test set accuracy of 0.95

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture was Lenet, because the images of traffic sign are just grayscale, and information is also not too much.


* What were some problems with the initial architecture?

The input shape in the initial architecture was 32x32x3, and was changed to 32x32x1 due to grayscale. What's more, the last full connection layer was changed to 43 because the traffic sign has 43 class.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

If under fitting, we can try to add serval convolution layers to abstract more information from images;
If over fitting, we can try pooling, dropout or regularization to make model less complex. 


* Which parameters were tuned? How were they adjusted and why?

The batch size and learning rate are tuned. The batch size can speed up training and also boost performance. The suitable learning rate can make the training stable and also boost performance.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The convolution works well because it can abstract multi-information from the images, including geometry, colors, shapes, and so on. The dropout layer can help createing a successful model, especially complex model, because it can reduce computation and overcome overfitting.

If a well known architecture was chosen:
* What architecture was chosen?

LeNet


* Why did you believe it would be relevant to the traffic sign application?

Because the images of traffic sign are just grayscale, and information is also not too much.


* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The training set accuracy is 0.8, and the testing set accuracy is 0.95, the validation set is not used.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory      		| Roundabout mandatory   									| 
| Speed limit (50km/h)     		| Speed limit (50km/h) 										|
| Yield					| Yield											|
| Keep right	      			| Keep right					 				|
| Speed limit (70km/h)			| Speed limit (70km/h)      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Roundabout mandatory   									| 
| 1.0    			| Speed limit (50km/h) 										|
| 1.0				| Yield											|
| 1.0	      			| Keep right					 				|
| 1.0				| Speed limit (70km/h)      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


