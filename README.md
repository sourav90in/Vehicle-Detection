## Writeup:
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/Car.png
[image2]: ./examples/NotCar.png
[image3]: ./examples/test1.png
[image4]: ./examples/CH_Ytest1.png
[image5]: ./examples/CH_Crtest1.png
[image6]: ./examples/CH_Cbtest1.png
[image7]: ./scaled_wdw_img/test1.jpg

[image8]: ./hog_img/CH_Ytest4.png
[image9]: ./hog_img/CH_Crtest4.png
[image10]: ./hog_img/CH_Cbtest4.png
[image11]: ./scaled_wdw_img/test4.jpg
[image12]: ./labels/test4.jpg
[image13]: ./final_boxed_img/test4.jpg

[image14]: ./examples/labels_vid/296.png
[image15]: ./examples/labels_vid/297.png
[image16]: ./examples/labels_vid/298.png
[image17]: ./examples/labels_vid/299.png
[image18]: ./examples/labels_vid/300.png
[image19]: ./examples/labels_vid/301.png

[image20]: ./examples/one_f_boxes/296.jpg
[image21]: ./examples/one_f_boxes/297.jpg
[image22]: ./examples/one_f_boxes/298.jpg
[image23]: ./examples/one_f_boxes/299.jpg
[image24]: ./examples/one_f_boxes/300.jpg
[image25]: ./examples/one_f_boxes/301.jpg

[image26]: ./examples/labels_agg/296.png
[image27]: ./examples/labels_agg/297.png
[image28]: ./examples/labels_agg/298.png
[image29]: ./examples/labels_agg/299.png
[image30]: ./examples/labels_agg/300.png
[image31]: ./examples/labels_agg/301.png

[image32]: ./examples/res_boxes/296.jpg
[image33]: ./examples/res_boxes/297.jpg
[image34]: ./examples/res_boxes/298.jpg
[image35]: ./examples/res_boxes/299.jpg
[image36]: ./examples/res_boxes/300.jpg
[image37]: ./examples/res_boxes/301.jpg

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The first step was reading in all the images provided for vehicles and non-vehicles in the data-sets: https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip and https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

Since the images in the Vehicle database were from multiple sources like GTI-Left, GTI-Middle etc and each such folders had a time-sequential nature of data collection, I shuffled all the image names across a consolidated list of all images sources before reading in each image. Alongside reading all the provided images, I also created their corresponding labels i.e. 1 for Vehicle and 0 for Non-Vehicle.

One example each of the 'vehicle' and 'non-vehicle' classes is:

![Vehicle][image1]
![Non-Vehicle][image2]

Further once all the images were read, the training was performed in the function train_data() which is responsible for Feature Extraction/Scaing/Training and saving the results to a pickle file.

This function train_data() does the following:
a) For all the images loaded in previous step, they were converted to YCrCb colour space and HOG features were extracted
for them for all the 3 channels Y,Cr,Cb sequentially by calling get_hog_features()
b) Spatial Binned and Colour Histogram features were also extracted by calling bin_spatial() and color_hist() respectively
c) All the extracted features are ravelled and horizontally stacked and appended to a feature list.
d) Once feature extraction is completed for the entire set of images, the Feature-set is split into training and test-set images.
e) The Standard Scaler is fit on the Training feature set and is then used to transform the Training and Test feature sets.
f) The scaled features are then fed into a LinearSVC which performs the training and the accuracy score is then predicted for the test-set.
Eventually all the parameters chosen for the Feature Extraction as well as the fitted Scaler as well as the Trained Classifier are stored in a pickle file for ease of usage in other stages.

#### 2. Explain how you settled on your final choice of HOG parameters, Sptatial binned parameters and other parameters

In order to finalize the HOG parameters( orientations, pixels_per_cell,cells_per_block), Sptial size for spatial binned features and number of bins for Colour Historgram features, I did not try to observe its impact on the test-set obtained from the training and test-set split of the data-set provided which only contained a Veicle or a patch of the road . It was difficult to understand the ideal combination of parameters needed for detection of vehicles present in a frame where other objects were present as well.

I rather tried to observe its impact on performance on a real-world frame similar to those present in the test-images folder.

I followed the below steps:

a) First I kept the default value of orientations at 9, hist_bins at 32 and cell_per_block at 2 and selected multiple scales ranging from 0.65 to 2.15 with an increment of 0.10.

b) I altered the pix_per_cell and spatial_size gradually from 8 to 20 and 32 to 16 respectively and performed the Training as well as Window detection on the images prsent in the test-images folder.

I tried the above steps a) and b) with two different colour-space conversions i.e. RGB to YCrCb and RGB to LUV and ran the executed the rudimentary implementation of find_cars function(which extracts HOG features by HOG sub-sampling) provided in the lessons to obtain an estimate of performance.

With the above steps, I found that with the final choice of pix_per_cell as 16 and spatial_size as 16 and the color space as YCrCb, the performance of detected vehicles in image-frames was quite good and better than using the LUV color-space.

Ahtough the number of overlapping detected windows returned from find_cars was quite a huge number but the clustering of most windows were aroudn the actual Vehicles which prompted me to finalize these parameter values and use them for Stand-alone image processing as well as processing of Video Frames.
(P.S.: Eventually I modified the implementation of find_cars to find_cars_v3 which performs a Union of Windows detected across all scales in the ROI of the image and returns only one Hot window per detected object as well as reduced the number of scales used to search across image frames.)

Also accuracy score on the test-set obtained from the training and test-set split still remained above 99% across these parameter variations.

Some of the results obtained for the test1.png file are as follows:

![Test-Image][image3]
![HOG Y Channel][image4]
![HOG Cr Channel][image5]
![HOG Cb Channel][image6]
![Detected Windows][image7]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As indicated above, the feature extraction as well as the Training of the classifier was performed in train_data().

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The Sliding Window search has been implemented in find_cars_v3 which re-uses most of the code provided in the HOG Sub-Sampling method introduced in the lessons with a few modifications. Also I did not modify the Window Overlap factor in terms of cells_per_step and used the same value of 2 that was used in the provided code. The modifications made to the find_cars function are :

a) It can scan over muliple scales now.
b) It rejects detected windows whose prediction is that of a vehicle but has a decision-score lower than 1.1
c) It aggregates the detected windows that satisfy criteria of step b) over all scales and then calculates the heat-map for the entire set of aggregated windows.
d) It calculates labels on the aggregated heat-map and then returns the bounding boxes for the objects classified as non-zero in the output of labels.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In the initial implementation, I tried to carry out a GridSearch for deriving the optimal values of C and gamma for the LinearSVC across the features of the training set but it took forever to retrieve the optimal values. As a result of the huge time-latency, I decided to bypass the GridSearch and went ahead with the default set of parameters for the LinearSVC SVM classifier.

The Pipeline for any specific Image input works as follows for any set of input scales:
a) It converts the image to YCrCb colour space.
b) It extracts HOG features,Spatially Binned and Colour Histogram features and horizontally stacks them.
c) It scales the feature-set by the fitted Scaler obtained from Training.
d) It performs a prediction on the Scaled Feature-set.
e) If the decision score of the prediction is greater than 1.1, this detected window is retained.

All the aggregated windows across scales are subjected to Heat-Map aggregation across scales for any frame. This has an effect of performing a union of all detected windows across multiple scales and returning just one window per detected object.

Here are the different stages of the Pipiline on test-image4:

HOG-Channel Features:

![HOG_ChannelY][image8]
![HOG_ChannelCr][image9]
![HOG_ChannelCb][image10]


Windows that have a decision score of greater than 1.1 across all scales super-imposed on original image:

![All_Windows][image11]

Labels generated after Heat-Map aggregation across detected windows for all scales:

![Labels][image12]


Final Bounding boxes overlayed on top of original image:

![Final Boxes][image13]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

False Positives were removed by three methods:

1) Dropping some of the detected windows which have an absolute decision score of less than 1.1.

2) Heat-Map aggregation across multiple scales for a frame as well as across frames of a video as explained below.

3) The Region of interest per frame was narrowed down to the lower half of the image as well as the starting X-coordinate per frame of the image was set to 250. This speeded up the feature extraction as well as avoided false detections in the sky or the other-side of the road where cars were present too.


Overall in the entire project, I have utilized two levels of Heat-Map aggregation:

a) First level of Heat-Map aggregation is for detected windows across all scales for a specific frame. This has an effect of performing a union of all detected windows across multiple scales and returning just one window per detected object. This is done in the find_cars_v3() function.

b) Second level of Heat-Map aggregation is for detected windows across the last 50 frames of the video, followed by thresholding to regions in the aggregated heat-map with a threshold of 8 windows. This has a smoothening effect on the bounding boxes determined per frame of the video as well as helps in removing quite a lot of False Positives. This is implemented in the member functions: append_boxes and get_heat of the class BoxTracker().


### Here are Labelled heatmaps of 6 frames of the video:

![Video Frame 296][image14]
![Video Frame 297][image15]
![Video Frame 298][image16]
![Video Frame 299][image17]
![Video Frame 300][image18]
![Video Frame 301][image19]

### Here are Bounding boxes determined per frame(without aggregation across frames):

![Video Frame 296][image20]
![Video Frame 297][image21]
![Video Frame 298][image22]
![Video Frame 299][image23]
![Video Frame 300][image24]
![Video Frame 301][image25]


### Here is the output of the integrated heatmap for these 6 frames across 50 frames prior to it:
![Video Frame 296][image26]
![Video Frame 297][image27]
![Video Frame 298][image28]
![Video Frame 299][image29]
![Video Frame 300][image30]
![Video Frame 301][image31]


### Here the resulting bounding boxes which are drawn onto these 6 frames:
![Video Frame 296][image31]
![Video Frame 297][image33]
![Video Frame 298][image34]
![Video Frame 299][image35]
![Video Frame 300][image36]
![Video Frame 301][image37]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue faced was with respect to obtaining a balance between good results and time taken for Feature Extraction per frame. I had to reduce the number of scales used in the final implementation for faster processing.I also came across the OpenCV implementation of HOG which is supposedly faster but could not fit it into the current structure of the other written functions such as find_cars_v3, so I avoided its usage.

The other problems faced wwere with respect to wobbly boxes and boxes getting split quickly across frames of the video. Using the BoxTracker class and its methods to aggregate detected boxes across frames as well as aggregation of detected windows across multiple scales, eventually helped in addressing this problem.

Another issue was with respect to cars getting detected on the other side of the road which even though accurate was not desirable, so having a starting X-Coordinate of 250 for the targeted ROI helped in mitigating this issue. However this has a huge chance of failure for more curvy roads as well as it can also fail for cases where there are cars in the same lane as the car from which the video is taken.

To make it more robust I can potentially try out the You Only Look Once Deep Neural Approach to solving this project as a future extension of this problem.

