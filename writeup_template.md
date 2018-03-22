##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image2]: ./examples/HOG_example.jpg
[image3]: ./heatmaps.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]


I noticed a small discrepency between performance on the training set and performance in the actual video. For instance, HLS color space got the best results on the training set; however, the the classifier trained on hog features from HLS images tended to only recognize certain parts of the car--the taillights and the door handles.  The YCrCb classfier did not do as well on the training set but produced nice bounding boxes around cars in images.  The YCrCb classfier is less accurate than the HLS classifier, but it is more scale invariant.  I did not have time to optimize the other parameters on the road images..  

####2. Explain how you settled on your final choice of HOG parameters.

I optimized the HOG parameters on the training set.  I considered doing an exhaustive grid search using sklearn's model_selection module.  However, I had developed a pretty good idea of what the optimal parameters would be from the in-class exercises, so I decided a formal method wasn't necessary.  Instead I just did a little bit of tweaking and observed trends in performance to find the right parameters.  As mentioned previously HLS performed best on the training set but had issues on the road images, so I went with YCrCb.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM on the train and test images.  The complete training code is in the 3rd cell block.  I used both color and oriented gradients to train the classifier.  The final accuracy was around 98%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched from where the horizon usually appeared in the y-axis (around 400) to nearly the bottom of the image.  I used maximum overlap of cells to achieve the best performance.

I tried a lot of different scales ranging from .4 to 1.4.  I think I flipped the scale from the class implementation--in my code the scale is applied to the image so .5 means the image is half the size, so the window is twice as big.  I eventually settled on scales ranging from .6 to 1.1, as these provided the most useful information.

The code is implemented in the "slide_window_over_feats" function and the "find_cars" function.

####2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

I pretty exhaustively tweaked the threshold value for the genereate_heatmap function of the heatmap class.  Images are in the jupyter notebook. 
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

I ignore any boxes that are completely overlapped by a larger bounding box.  Partial overlap seems justified as nearby cars may partially overlab, but should not be assigned to the same box.

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had to make my heatmap threshold very high and keep many frames to avoid false positives.  Although I reduced false positives, I also added considerable lag to my detection pipeline.  The detection pipeline doesn't do well for cars that quickly zip by in the opposite direction.  It struggles with cars in the same line that are moving much faster or much slower than the camera.

I think a deep learning solution like YOLO or an SSD would do better than the hog-svm pipeline.

