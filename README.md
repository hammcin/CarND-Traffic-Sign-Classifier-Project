# Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Figures/train_class_hist.jpg "Train Class Histogram"
[image2]: ./Figures/valid_class_hist.jpg "Validation Class Histogram"
[image3]: ./Figures/test_class_hist.jpg "Test Class Histogram"
[image4]: ./Figures/aug_train_class_hist.jpg "Augmented Train Class Histogram"
[image5]: ./Figures/grayscale.jpg "Grayscaling"
[image6]: ./Figures/data_aug.jpg "Augmented Data"
[image7]: ./Figures/bicycles-crossing.jpg "Bicycles Crossing Traffic Sign"
[image8]: ./Figures/no-entry.jpg "No Entry Traffic Sign"
[image9]: ./Figures/priority-road.jpg "Priority Road Traffic Sign"
[image10]: ./Figures/roundabout-mandatory.jpg "Roundabout Mandatory Traffic Sign"
[image11]: ./Figures/yield.jpg "Yield Traffic Sign"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use the template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](./Traffic_Sign_Classifier.ipynb).

If you have trouble with the above link, here is a link to my [project code](./Traffic_Sign_Classifier.pdf) as a pdf.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The first figure is a histogram of the class labels for the training set:

![alt text][image1]

The next figure is a histogram of the class labels for the validation set:

![alt text][image2]

The last figure is a histogram of the class labels for the test set:

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I decided to generate additional data because this was done in the published baseline model on this problem (Sermanet, P. and LeCun, Y. Traffic Sign Recognition with Multi-Scale Convolutional Networks.).

The code I used for my data augmentation came from [here](https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9).

I also performed data augmentation because the best way to improve accuracy is to use more data to train the model.  An easy way to generate more data is to perform augmentation using such tranformations as scaling, translation, and rotation because the model is intrinsically invariant to these transformations.

To add more data to the the data set, I used the following techniques: scaling ([.9,1.1] ratio), translation ([-2,2] pixels for both x- and y- directions), and rotation ([-15,+15] degrees) because the model is intrinsically invariant to these transformations.

Here are examples of some of the original images from that training set and some augmented images:

![alt text][image6]

The difference between the original data set and the augmented data set is the following: I generated 5 additional data sets from the original training set using random scaling, translation, and rotation.  After data augmentation, I had a final total of 208794 images for my training set.

The difference that data augmentation made for the number of images available for training can be seen by comparing the original histogram for class labels in the training set with the new histogram for class labels in the training set after augmentation:

![alt text][image4]

Next, I decided to convert the images to grayscale because in the published baseline model on this problem (Sermanet, P. and LeCun, Y. Traffic Sign Recognition with Multi-Scale Convolutional Networks.), grayscaling the images improved the accuracy of the classifications.  The reason for the improvement may be that grayscaling simplifies the data so that it is easier for the model to learn.

Here is an example of a traffic sign image before and after grayscaling:

![alt text][image5]

As a last step, I normalized the image data because to improve the numerical stability of the calculations in the model and so that the classification is a well-conditioned problem.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		    |     Description	        					            |
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x1 Grayscale image  						          |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	  |
| RELU					        |												                        |
| Max pooling	      	  | 2x2 stride,  outputs 14x14x6 					        |
| Convolution 5x5	      | 1x1 stride, valid padding, outputs 10x10x16  	|
| RELU					        |												                        |
| Max pooling	      	  | 2x2 stride,  outputs 5x5x16 					        |
| Fully connected		    | Input = 400, Output = 120        				      |
| RELU					        |												                        |
| Dropout				        | keep probability = 0.5 						            |
| Fully connected		    | Input = 120, Output = 84        				      |
| RELU					        |												                        |
| Dropout				        | keep probability = 0.5 						            |
| Fully connected		    | Input = 84, Output = 43        				        |
| Softmax				        |												                        |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer, a batch size of 128, 50 epochs, a learning rate of 0.001, and a dropout probability of 0.5.  Additionally, I initialized my model with a truncated normal distribution with mean = 0 and standard deviation = 0.1.  I initialized the biases to 0.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.975
* test set accuracy of 0.959

I chose to use the LeNet architecture.  I chose this architecture because it performs well on MNIST.  When I first tried LeNet, I observed that the accuracy on the training set was high, but the accuracy on the validation set was low.  This led me to believe the model architecture was doing a good job of learning the data, but that the data set I was using was not the best.

At this point, I decided to learn more about the published baseline model on this problem (Sermanet, P. and LeCun, Y. Traffic Sign Recognition with Multi-Scale Convolutional Networks.).  The easiest change that I thought of that I could make from reading the paper was to change from using RGB images to grayscale images.  This resulted in a large gain in accuracy on the validation set.  The validation set accuracy was high, but not as high as I needed it.  One important lesson I learned from grayscaling the images was that I needed to grayscale the images first and then normalize the images, in that order.

The next difference I noticed in my model and the published baseline model was that the baseline model employed data augmentation.  I incorporated the same data augmentation used in the baseline model into my model (scaling, translation, and rotation).  Making this change improved my accuracy on the validation set to where I needed it to be.

The last change I made was to incorporate dropout layers, train for a large number of epochs (50) and use early stopping.  Making this last change got the accuracy of my model to its final value.

From my experience working on this problem, the most important changes that I made to my pipeline that made the biggest differences in my accuracy were grayscaling the images and performing data augmentation.

The values of my accuracy on the training and validation sets are both high, so I do not believe my model is either over- or under-fitting.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]

The images I collected from the web are all very clear images of the signs.  The lighting conditions appear to be good in all the images with the exception of, possibly, the first sign, which is a Priority Road sign.  All the images are cropped so that the traffic sign is in the center of the image, without much background.  Problems may arise with the second (Yield sign) and fourth (Roundabout Mandatory sign) images due to perspective transforms.  Also, the fourth image (Roundabout Mandatory sign) may blend in some with the blue sky behind it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			            |     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| Priority Road    		  | Priority Road   								              |
| Yield     			      | Yield 										                    |
| No Entry				      | No Entry										                  |
| Roundabout Mandatory	| Right-of-way at the Next Intersection			    |
| Bicycles Crossing		  | Priority Road      								            |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This does not compare favorably to the accuracy on the test set of 0.959.

I believe the reason my classifier accurately predicted the classes: Priority Road, Yield, and No Entry, was because of the number of examples available for training the classifier on these classes: 11340, 11520, 5940, respectively.  In contrast, the classifier did not not accurately predict the classes: Roundabout Mandatory and Bicycles Crossing, where the number of training examples were 1800 and 1440, respectively.  It seems that the number of examples available for training had a strong impact on the accuracy of the model for those classes.

| Class Label | Number of Training Examples | Training Accuracy |
|:-----------:|:---------------------------:|:-----------------:|
| 12          | 11340                       | 0.976             |
| 13          | 11520                       | 0.994             |
| 17          | 5940                        | 0.985             |
| 40          | 1800                        | 0.873             |
| 29          | 1440                        | 0.944             |

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 63rd cell of the Ipython notebook.

For the first image, the model is not sure about its prediction that this is a Road Work sign (probability of 0.40), and the image contains a Bicycles Crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| 0.68         			    | Priority Road   									            |
| 0.21     				      | Ahead Only 							                      |
| 0.08					        | Turn Left Ahead								                |
| 0.00	      			    | Right-of-way at the Next Intersection					|
| 0.00				          | Vehicles Over 3.5 Metric Tons Prohibited			|

For the second image, the model is completely sure that this is a No Entry sign (probability of 1.00), and the image does contain a No Entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| 1.00         			    | No Entry   									                  |
| 0.00     				      | Speed Limit (20km/h) 								          |
| 0.00					        | Speed Limit (30km/h)									        |
| 0.00	      			    | Speed Limit (50km/h)					 						    |
| 0.00				          | Speed Limit (60km/h)			                    |

For the third image, the model is completely sure that this is a Priority Road sign (probability of 1.00), and the image does contain a Priority Road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| 1.00         			    | Priority Road   								              |
| 0.00     				      | No Entry 							                        |
| 0.00					        | Roundabout Mandatory											    |
| 0.00	      			    | No Passing					 				                  |
| 0.00				          | Speed Limit (20km/h)							            |

For the fourth image, the model is not sure about its prediction that this is a Right-of-way at the Next Intersection sign (probability of 0.49), and the image contains a Roundabout Mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| 0.39         			    | Right-of-way at the Next Intersection   		  |
| 0.16     				      | Turn Left Ahead 									            |
| 0.11					        | Speed Limit (60km/h)									        |
| 0.07	      			    | Children Crossing					 		                |
| 0.05				          | Ahead Only					                          |

For the fifth image, the model is completely sure that this is a Yield sign (probability of 1.00), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| 1.00         			    | Yield   										                  |
| 0.00     				      | Speed Limit (20km/h) 							            |
| 0.00					        | Speed Limit (30km/h)							            |
| 0.00	      			    | Speed Limit (50km/h)					 		            |
| 0.00				          | Speed Limit (60km/h)							            |
