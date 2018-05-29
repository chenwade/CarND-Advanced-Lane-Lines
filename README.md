# Self-Driving Car Engineer Nanodegree Program
## Advanced Lane Finding Project

Sorry, this README isn't the final version. It will be improved in these few days.
The goals / steps of this project are the following:

* Caculate the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images based on the parameters computed above.
* Use color threshold, canny, sobel, etc., to create a thresholded binary image.
* Use hough line method to help to find the source points, so that we can get the projection matrix without given it in advanced. Then the image can be transformed from perspective view to the "birds-eye view".
* Detect lane pixels and fit to find the lanes.
* Sanity check for verifying the detected lanes are corrects are not.
* Adjust the lanes based on the sanity check and previous correct results.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Roughly evaluate the performance of the algorithm. 

[//]: # (Image References)

[im01]: ./output_images/calibrate_before.jpg "Chessboard image"
[im02]: ./output_images/corners_found.jpg "Corner found chessboard image"
[im03]: ./output_images/calibrate_after.jpg "Undistorted Chessboard"
[im04]: ./output_images/undistored.jpg "Undistorted Image"
[im05]: ./output_images/edged_try.jpg "Information filtered Image"
[im06]: ./output_images/edged.jpg "Edged Image"
[im07]: ./output_images/warped.jpg "warped image"
[im08]: ./output_images/lane_fit.jpg "Lane fit image"
[im09]: ./output_images/annotated.jpg "annotated image"
[im10]: ./output_images/debug.jpg "debug image"


##Files:
* main.py: the main code to start the pipeline
* lane.py: the most important part of code, it defines a Lane class which collect the lane information and contains the methods such as detect/lit lane line, sanity check and so on.
 * extract_lane_info.py: the code for extracting the edge and color .. information from the image.
 * transform_perspective.py: the code for tranforming the perspective view to bird's eye view
* calibrate_camera.py: the code for calibrating the camera
* my_datastruct.py: I write a circule queue for recording lastn lane fit information  
* video_helper.py: the code for processing the video
* evaluate_algorithm.py: contain a function to roughly evaluate the performance of the algorithm

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The OpenCV functions `findChessboardCorners` and `calibrateCamera` are the backbone of the image calibration. A number of images of a chessboard, taken from different angles with the same camera, comprise the input. Arrays of object points, corresponding to the location (essentially indices) of internal corners of a chessboard, and image points, the pixel locations of the internal chessboard corners determined by `findChessboardCorners`, are fed to `calibrateCamera` which returns camera calibration and distortion coefficients. These can then be used by the OpenCV `undistort` function to undo the effects of distortion on any image produced by the same camera. Generally, these coefficients will not change for a given camera (and lens). The below image depicts the corners drawn onto one chessboard image using the OpenCV function `drawChessboardCorners`:

Chessboard image:
![alt text][im01]
Corner found chessboard image:
![alt text][im02]

0The image below depicts the results of applying `undistort`, using the calibration and distortion coefficients:

![alt text][im03]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The image below depicts the results of applying `undistort` to one of the images:

![alt text][im04]

The effect of `undistort` is subtle, but can be perceived from the difference in shape of the hood of the car at the bottom corners of the image.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I explored several combinations of sobel gradient thresholds and color channel thresholds in multiple color spaces.  

![alt text][im05]

Ultimately, I use both sobelx and sobely to obtain the graident information, both hsv and rgb color space to detect white and yellow region. By the way, region of interest was also used to filter useless information. 
Below are the results of applying the binary thresholding pipeline to various sample images:
![alt text][im06]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.


 The `unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  Here I only hardcode the destination points in the pipeline. First, I use the mutiple version of hough tranform to find the straight left and right lines in the image. Then choosing 2 points in each line to consist the ('src') points. The detail is in the get_projection_matrix() of lane.py.

 The image below demonstrates the results of the perspective transform: 

![alt text][im07]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions `fit_lane_line` and `tune_lane_line`, which identify lane lines and fit a second order polynomial to both right and left lane lines. The first of these computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. Originally these locations were identified from the local maxima of the left and right halves of the histogram. The function then identifies image height / 10 windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image. Pixels belonging to each lane line are identified and the Numpy `polyfit()` method fits a second order polynomial to each set of pixels. The image below demonstrates how this process works:

![alt text][im08]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is based upon [this website](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) and calculated in the code cell titled "Radius of Curvature and Distance from Lane Center Calculation" using this line of code (altered for clarity):
```
curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```
In this example, `fit[0]` is the first coefficient (the y-squared coefficient) of the second order polynomial fit, and `fit[1]` is the second (y) coefficient. `y_0` is the y position within the image upon which the curvature calculation is based (the bottom-most y - the position of the car in the image - was chosen). `y_meters_per_pixel` is the factor used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters. 

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:
```
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * x_meters_per_pix
```
`r_fit_x_int` and `l_fit_x_int` are the x-intercepts of the right and left fits, respectively. This requires evaluating the fit at the maximum y value (719, in this case - the bottom of the image) because the minimum y value is actually at the top (otherwise, the constant coefficient of each fit would have sufficed). The car position is the difference between these intercept points and the image midpoint (assuming that the camera is mounted at the center of the vehicle).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][im09]

#### 7. Sanity check mechanism.

The class Lane provides some functions for sanity check, including:
1. whether the two lanes are rough parallel 
2. whether the fit lanes is similar to the last fit lanes 
3. whether the left lane has similar radius of curvature to the right lane
4. add the detect fit lane into the recentN_detect, which is a circular queue to store the recent fit lane information.
If the detected lanes pass the three above checks, we can think that we find the correct lane. And the consecutive detect frame number will add one. Or, the consecutive detect frame number will reset to zero and the consecutive not detect frame number will add 1.

In order to get better results, we made some rules that we can adjust the lane based on the checking result.
1. if we failed to pass sanity check in 5 consecutive frames, return False, it means we aren't going to annotate lane on the image
2. if both left_fit and right_fit valid ,we don't adjust the detect fit.
3. if left_fit valid but right_fit failed ,we adjust the right_fit based on
   a.the latest right_fit which passed the sanity check
   b. the valid left_fit
4. if left_fit valid but right_fit failed ,we adjust the right_fit based on
    a.the latest right_fit which passed the sanity check
    b. the valid left_fit
5. if neither left_fit or right_fit valid, we adjust two fit based on
    a.the latest right_fit which passed the sanity check
 
#### 8. Debug mechanism.               
Here I define a debug_manager class to help us debug. It can show lots of important information in a window. The image below is an example:

![alt text][im10]


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's the annotated project video [link to my video result](./project_video_out.mp4)
Here's the annotated challenge video [link to my video result](./challenge_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In order to know the details of each step in the algorithm, I use a DebugManager class to help record the debug information. And it is also easy to debug the video.
Annotating a whole video cost lots of time. So I also write a video_helper.py to process the video. It include the function of getting the subset of video and so on.
In evaluate_algorithm.py, it provide a rough evaluation method for lane detecting algorithm.

[Project video - debug version](./project_debug_out.mp4)

[Challenge video - debug version](./challenge_debug_out.mp4)

###Acknowledge

* Udacity
* https://github.com/diyjac/SDC-P4
