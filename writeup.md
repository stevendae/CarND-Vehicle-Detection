## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./examples/car_not_car.png
[image2]: ./examples/features1.png
[image3]: ./examples/BoundingBox.png
[image4]: ./examples/window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fourth code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of all the features that were used for vehicle detection. They include a compressed 16x16 spatial image, a color histogram using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and sought experiential support from the Udacity forums to finalize the HOG parameters. 

I started my experimentation with my initial conditions set to that of the default parameters that were used in the lessons.
These parameters included:

Colourspace: HLS
Orientation: 9
Pixels Per Cell: 8,8)
Cells Per Block: (2,2)
Hog Channel: 2

I then further tweaked my parameters to (1st Submission):

Colourspace: YCrCb
Orientation: 9
Pixels Per Cell: (8,8)
Cells Per Block: (2,2)
Hog Channel(s): ALL

and finally settled on (2nd Submission):

Colourspace: YUV
Orientation: 9
Pixels Per Cell: (8,8)
Cells Per Block: (2,2)
Hog Channel(s): ALL
Spatial Size: (16,16)
Hist Bins: 16

The changes to colourspace were made based on the anecdotal evidence floating in the Udacity forums around the notion that YUV had improved accuracies in the test sets. The YUV space is characterized by the three components whereby the Y component is the luminance component (the brightness) and U and V are the chrominance (color) components. This means that YUV represents color as brightness and two color difference signals. The representation of YUV separates the luminance and chrominance, so the computing system can encode the image in a way that less bits are allocated for chrominance. This is done through color subsampling, which simply encodes chrominance components with lower resolution. 

The changes to the number of HOG channels was on the basis that when more information is provided to represent a class, the more distributed the data sets will become and hence a better generalization as a result. 

The other parameters were left unchanged by virtue of a good enough result that was achieved from it remaining as is. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM clasifier using three features from each image. The first feature was the compressed spatial colour vector where the individual pixels from a single channel of a compressed image were flattened into a one dimensional array. Each spatial colour vector from each channel was concatenated into one long spatial colour vector. This section can be found in the fifth code cell of the IPython notebook.  

The second feature was the colour histogram which organized each pixel of the channel of the image into the closest colour bin wherby the colour bins were separated into 32 equally sized bins from a range of 0 to 255. Each vector of length 32 that represented the color histogram of the channel of the image was concatenated for all three vectors into a single vector of length 96. This section can be found in the sixth code cell of the IPython notebook.

The third and last feature was the HOG feature where each channel had produced it's own vector that represented the quantities in each bin of the HOG channel. A final vector was created by concatenating all three vectors into one. This can be found in the third code cell of the IPython notebook.

In order to create a represetative vector of each image, all three features were appended together and then concatenated into a single multi-dimensional vector. This section is described in the 9th code cell of the IPython notebook under the title 'Feature Extraction'.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search in the 16th code cell of the IPython notebook. This was achieved by implementing a subsampling algorithm where the HOG features were extracted for the entire desired region of the image and then spatial colour, colour histogram, and HOG features are extracted from each sample window. This method is the same procedure that is prescribed in the lessons. The decision for the scales to search was based on testing what scaling ranges provided correct detections. It was found that scaling ranges between 1 and 2 with increments of 0.25 provided a good range to detect cars of various distance which is translated at the image into size. The overlap ratio was chosen as 75% which resulted in a cells per step value of 2. This was so that the sub sampled windows can cover as much of the region of interest as possible. 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

I included the Udacity image data set as that provided more data so that recognition of the white vehicles was significantly more robust. 
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. These positions characterized the coordinates of the bounding boxes of positive detections in the frame. The bounding box coordinates were then added to a tensor that stores the bounding box coordinates of the last three frames. These historical coordinates are concatenated into a single vector to depict bounding boxes of the last three frames which is projected onto the current frame. From the positive detections of the last three frames projected onto the current frame I created a heatmap and then thresholded that map to identify vehicle positions and eliminate false positives. I added the heatmap into another tensor that stores the heatmap coordinates of the last five frames and then implemented a threshold of the average of these five frames to further false positives. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.


Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The approach I took was the general procedure that was illustrated in the Udacity lessons. I extracted spatial, colour, and HOG features and used them to train a SVM classifier. I then used a sliding window technique to predict if an area of the frame contained a positive detection. I collected positive detections for the last three frames in order to supplement my positive detections and ensure that a true positive detection was in the frame via consistency in real space. I then applied a heat map and threshold to add another level of filtering. 

My strategy for success in this project was to ensure that I satisfied a minimal requirement at each stage. The first stage was to see if my classifier was predicting the vehicles at many scales. Once this was established I made a collections tensor so that I can implement a high threshold on the heat map and 'track' actual positive detections. I then collected heat map values of the last five frames and then averaged them to add another confidence filter. This method was derived as a result of finding ways to truly eliminate unwanted false positives in the video clip which happened to be the pervasive issue that kept arising.

My pipeline may fail during different lighting/weather/daytime conditions based on the fact that two thirds of the feature is based on the color and pixel information of the training images. Therefore the SVM model will have to be tested on different video styles.

There is also some concern with the rate of predictions of using an SVM to perform vehicle detection. I would need to further optimize the HOG and SVM for additional real-time performance. 
