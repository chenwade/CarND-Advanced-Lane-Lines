import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from debug_manager import *
from transform_perspective import *
from image_preprocess import *
from calibrate_camera import *


class MyCirculateQueue(object):
    """
    a customize circular queue based on the python list

    this queue is used for storing recent lane fit information
    """
    def __init__(self, maxsize):
        # define the max length of queue
        self.maxsize = maxsize
        # the queue
        self.queue = []

    def __str__(self):
        # return the str of the object
        return str(self.queue)

    def size(self):
        # the number of items in the queue
        return len(self.queue)

    def is_empty(self):
        # return True if nothing in self.queue
        return self.queue == []

    def is_full(self):
        # return True if queue if full
        return self.size() == self.maxsize

    def enqueue(self, item):
        # enqueue a item
        if self.is_full():
            self.dequeue()
        self.queue.insert(0, item)

    def dequeue(self):
        # dequeue a item
        if self.is_empty():
            return None
        return self.queue.pop()

    def find(self, value):
        # if find value(content) in the queue, return the index
        # if not, return None
        for i in range(len(self.queue)):
            if self.queue[-1 - i] == value:
                return i
        return None

    def visit(self, index):
        # return the value(content) of the index of queue
        assert 0 <= index < len(self.queue)
        return self.queue[-1 - index]

    def get_tail(self):
        # get the last item which is enqueued
        if self.is_empty():
            return None
        return self.queue[0]

class Lanes(object):
    """
    The most important class for lane detection.
    It records useful parameter and info about the lane of image
    It contains multiple methods about lane fitting, sanity checking, verifying, annotating
    """
    def __init__(self, debug=False):
        # current frame number in video
        self.frame_num = 0
        # input/original image
        self.input_image = None
        # the height and width of image
        self.image_height = None
        self.image_width = None

        # current left lane fit by using fit_lane_line() or tune_lane_line()
        self.left_fit = np.array([False])
        # current left lane fit by using fit_lane_line() or tune_lane_line()
        self.right_fit = np.array([False])
        # current left lane fit adjusted by using adjust_anotated_lane()
        self.adjusted_left_fit = np.array([False])
        # current right lane fit adjusted by using adjust_anotated_lane()
        self.adjusted_right_fit = np.array([False])

        # the last n fits of the line
        self.left_recent_10fitted = MyCirculateQueue(10)
        self.right_recent_10fitted = MyCirculateQueue(10)

        # is fit reasonable?
        self.left_fit_valid = False
        self.right_fit_valid = False

        # is current lane detection reasonable?
        self.sanity_check_result = False
        # recent 10 lane detection results
        self.recent_10check = MyCirculateQueue(10)
        # the frame number that detect lane line consecutively
        self.con_detect_num = 0
        # the frame number that do not detect lane line consecutively
        self.con_not_detect_num = 0

        # annotate the lane in the area? (it depend on the sanity check result)
        self.annotate = False

        # for projection from perspective view to bird's eye view
        # perspective_src_points will be runtime computed
        self.perspective_src_points = None
        self.perspective_dst_points = np.float32(([384, 720], [896, 720], [896, 600], [384, 600]))
        self.M = None
        self.M_inv = None

        # lane_width and lane_length in pixel
        self.lane_width = 896-384
        self.lane_length = None

        # radius of curvature of the line in some units
        # suppose the initial lane is straight
        self.right_Roc = 10000.
        self.left_Roc = 10000.

        # distance in meters of vehicle center from the lane center
        self.vehicle_offset = 0.

        # if debug is True, initialize a debug manager
        self.debug = debug
        if self.debug is True:
            self.debug_manager = DebugManager()


        """
        Maybe used in the future: 
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        
        """



    """
        # for debug
    def visualize_fit_process(self, binary_warped):
        #####1
        binary_warped[(binary_warped > 0)] = 1
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        plt.plot(np.arange(0, binary_warped.shape[1], 1), histogram)
        plt.xlabel('width')
        plt.ylabel('pixel number')
        plt.title('the bottom half of the image')

        plt.grid()
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
        ####2
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 10
        # Set height of windows
        window_height = binary_warped.shape[0] // nwindows
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window * window_height)
            win_y_high = win_y_low - window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzerox > win_xleft_low) & (nonzerox < win_xleft_high)
                              & (nonzeroy < win_y_low) & (nonzeroy > win_y_high)).nonzero()[0]
            good_right_inds = ((nonzerox > win_xright_low) & (nonzerox < win_xright_high)
                               & (nonzeroy < win_y_low) & (nonzeroy > win_y_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # draw the points and fit line of the left and right lane
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        ax1.imshow(out_img)
        ax1.plot(left_fitx, ploty, color='yellow')
        ax1.plot(right_fitx, ploty, color='yellow')
        ax1.set_xlim(0, binary_warped.shape[1])
        ax1.set_ylim(binary_warped.shape[0], 0)

        ####3
        out_img1 = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img1)
        # Color in left and right line pixels
        out_img1[lefty, leftx] = [255, 0, 0]
        out_img1[righty, rightx] = [0, 0, 255]

        # Generate a polygon to illustrate the search window areas
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_(left_line_pts), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_(right_line_pts), (0, 255, 0))
        result = cv2.addWeighted(out_img1, 1, window_img, 0.3, 0)
        ax2.imshow(result)
        ax2.plot(left_fitx, ploty, color='yellow')
        ax2.plot(right_fitx, ploty, color='yellow')
        ax2.set_xlim(0, binary_warped.shape[1])
        ax2.set_ylim(binary_warped.shape[0], 0)
        plt.subplots_adjust()
        plt.show()
        return (left_fit, right_fit)
        """

    def has_similar_curvature(self, left_fit, right_fit):
        """
        We will compared the radius of left_fit and right_fit.
        The Roc calculated value varies, so we think most situations are reasonable, except:
        1.  0 < left_Roc < 2000 and -2000 < left_Roc < 0
        2.  0 < right_Roc < 2000 and -2000 < right_Roc < 0

        Parameters
        ----------
        left_fit: the quadratic fit of left lane
        right_fit: the quadratic fit of right lane

        Return
        ----------
        True or False

        """

        # calculate the Roc
        self.left_Roc = measure_radius_of_curvature(left_fit)
        self.right_Roc = measure_radius_of_curvature(right_fit)

        if self.left_Roc > 0 and self.left_Roc < 2000 and self.right_Roc > -2000 and self.right_Roc < 0:
            return False
        elif self.right_Roc > 0 and self.right_Roc < 2000 and self.left_Roc > -2000 and self.left_Roc < 0:
            return False
        else:
            return True

    def is_2lanes_parallel(self, left_fit, right_fit):
        """
        Check whether the left lane roughly parallel with right lane
        Check whether the distance between left lane and right lane reasonable

        Parameters
        ----------
        left_fit: the quadratic fit of left lane
        right_fit: the quadratic fit of right lane

        Return
        ----------
        check_result: the result of check, True or False
        """

        check_result = True

        # we record values every 10 pixels in height
        ploty = np.linspace(0, self.image_height - 1, self.image_height / 10)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # the distance between left fit and right fit, which can be regarded as a rough estimated lane width
        distances = right_fitx - left_fitx

        # diff is the difference between real lane width and two fit's distance
        diffs = distances - self.lane_width

        # set threshold
        threshold = 250

        for diff in diffs:
            if diff > threshold or diff < -threshold:
                check_result = False

        # mean square error
        mse = np.sqrt(np.sum(np.square(diffs)) / len(diffs))
        if mse > threshold:
            check_result = False

        # for debug manager collecting info
        if self.debug:
            self.debug_manager.parallel_mse = mse

        return check_result

    def is_2recent_fit_similar(self, current_fit, last_fit):
        """
        Check whether the fit lane roughly equal between two frames

        Parameters
        ----------
        current_fit: the quadratic fit of left/right lane in current frame
        last_fit: the quadratic fit of left/right lane in last frame

        Return
        ----------
        check_result: the result of check, True or False
        """

        check_result = True
        # for the first frame, the last_fit = None, so we assign the last_fit = [0 ,0, 0]
        if last_fit is None:
            last_fit = np.array([0, 0, 0])

        # we record a value every 10 pixels in height
        ploty = np.linspace(0, self.image_height - 1, self.image_height / 10)
        current_fitx = current_fit[0] * ploty ** 2 + current_fit[1] * ploty + current_fit[2]
        last_fitx = last_fit[0] * ploty ** 2 + last_fit[1] * ploty + last_fit[2]

        # the difference between current fit and last fit
        diffs = last_fitx - current_fitx

        #set threshold
        threshold = 70

        for diff in diffs:
            if diff > threshold or diff < - threshold:
                check_result = False

        #mean-square error
        mse = np.sqrt(np.sum(np.square(diffs)) / len(diffs))
        if mse > threshold / 2:
            check_result = False

        # for debug manager collecting info
        if self.debug:
            if self.left_fit is current_fit:
                self.debug_manager.left_fit_mse = mse
            elif self.right_fit is current_fit:
                self.debug_manager.right_fit_mse = mse

        return check_result


    def get_latest_valid_fit(self, which_lane):
        """
        find the latest valid left/right fit in recent 10 frames

        Parameters
        ----------
        which_lane: must be 'left' or 'right', 'left' means to get latest valid left fit
                    while 'right' means to get latest valid left fit

        Return
        ----------
        latest_valid_fit: latest valid fit
        """

        assert which_lane == 'left' or which_lane == 'right'
        # find the latest fit in recent 10 frame
        index = self.recent_10check.find(True)

        # all recent 10 frame lane fit isn't valid, return None
        if index is None:
            return None
        # get latest valid left fit
        if which_lane == 'left':
            latest_valid_fit = self.left_recent_10fitted.visit(index)
        # get latest valid right fit
        elif which_lane == 'right':
            latest_valid_fit = self.right_recent_10fitted.visit(index)
        return latest_valid_fit

    def predict_right_fit(self):
        """
         if left lane fit is valid but right lane fit doesn't valid, we can predict the right lane by using
         1. the recent valid right fit info
         2. the valid left fit info

         Parameters
         ----------

         Return
         ----------
         adjusted_right_fit: the adjusted right lane fit
         """

        # at first, we try to predict current right fit as the recent valid fit,
        # if there is valid fit in recent 10 frames
        latest_valid_right_fit = self.get_latest_valid_fit('right')
        # check is it match with the valid left lane fit?
        if (latest_valid_right_fit is not None) and self.is_2lanes_parallel(self.left_fit, latest_valid_right_fit) and \
                self.has_similar_curvature(self.left_fit, latest_valid_right_fit):
            adjusted_right_fit = latest_valid_right_fit
        else:
            # if no valid right lane fit in recent 10 frames or doesn't match with current valid left fit
            # simply predict the current right fit as the current valid left fit add lane width
            left_fit = self.left_fit
            left_fit[2] += self.lane_width
            adjusted_right_fit = left_fit
        return adjusted_right_fit

    def predict_left_fit(self):
        """
        if right lane fit is valid but left lane fit doesn't valid, we can predict the left lane by using
            1. the recent valid left fit info
            2. the valid right fit info

        Parameters
        ----------

        Return
        ----------
        adjusted_left_fit: the adjusted left lane fit
        """

        # at first, we try to predict current right fit as the recent valid fit,
        # if there is valid fit in recent 10 frames
        latest_valid_left_fit = self.get_latest_valid_fit('left')
        # check is it match with the valid right lane fit?
        if (latest_valid_left_fit is not None) and self.is_2lanes_parallel(latest_valid_left_fit, self.right_fit) and \
                self.has_similar_curvature(latest_valid_left_fit, self.right_fit):
            adjusted_left_fit = latest_valid_left_fit
        else:
            # if no valid left lane fit in recent 10 frames or doesn't match with current valid right fit
            # simply predict the current right fit as the current valid right fit minus lane width
            right_fit = self.right_fit
            right_fit[2] -= self.lane_width
            adjusted_left_fit = right_fit
        return adjusted_left_fit

    def predict_both_fits(self):
        """
        if neither left nor right lane fit is valid, we can predict them using the recent valid fit info

        Parameters
        ----------

        Return
        ----------
        adjusted_left_fit: the adjusted left lane fit
        adjusted_right_fit: the adjusted right lane fit
        """
        latest_valid_right_fit = self.get_latest_valid_fit('right')
        latest_valid_left_fit = self.get_latest_valid_fit('left')
        return latest_valid_left_fit, latest_valid_right_fit

    def get_projection_matrix(self, undistorted_img):
        """
        At first, use cascade hough line to get the source code and
        then compute the projection matrix for transform the perspective view to the bird's eye view

        Parameters
        ----------
        undistorted_img: the undistorted image

        Return
        ----------
        """
        # process the image, get the clear edge information by using color/graident and etc .. threshold methods)
        edged_image = image_preprocess2(undistorted_img)

        # cascading hough mapping line attempts
        hough = 1
        src_points = hough_lines1(edged_image)
        if src_points is None:  # exception happen in hough_lines1
            hough = 2
            src_points = hough_lines2(edged_image)
            if src_points is None:
                hough = 3
                src_points = hough_lines3(edged_image)
                if src_points is None:
                    hough = 4
                    src_points = hough_lines4(edged_image)

        assert src_points is None

        if src_points is not None:
            self.perspective_src_points = src_points

            # generate the projection matrix
            M, M_inv = transform_perspective(self.perspective_src_points, self.perspective_dst_points)
            self.M = M
            self.M_inv = M_inv


    def fit_lane_line(self, binary_warped):
        """

        * Calculate a histogram of the two thirds of the image
        * Partition the image into 10 horizontal windows
        * Starting from the bottom window, enclose a 200 pixel wide window around the left peak and right peak of the
                histogram (split the histogram in half vertically)
        * Go up the horizontal window slices to find pixels that are likely to be part of the left and right lanes,
                recentering the sliding windows opportunistically
        * Given 2 groups of pixels (left and right lane line candidate pixels), fit a 2nd order polynomial to each group
                , which represents the estimated left and right lane lines

        Parameters
        ----------
        binary_warped: the preprocessed and warped image

        Return
        ----------
        left_fit: the quadratic fit of left lane
        right_fit: the quadratic fit of right lane
        """

        binary_warped[(binary_warped > 0)] = 1
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[self.image_height//3:, :], axis=0)
        # plt.plot(histogram)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 10
        # Set height of windows
        window_height = binary_warped.shape[0] // nwindows
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        rectange_windows = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window * window_height)
            win_y_high = win_y_low - window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            #only for debug, record the rectange information
            rectange_windows.append([win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high])

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzerox > win_xleft_low) & (nonzerox < win_xleft_high)
                              & (nonzeroy < win_y_low) & (nonzeroy > win_y_high)).nonzero()[0]
            good_right_inds = ((nonzerox > win_xright_low) & (nonzerox < win_xright_high)
                               & (nonzeroy < win_y_low) & (nonzeroy > win_y_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.left_fit = left_fit
        self.right_fit = right_fit

        # for debug manager collecting info
        if self.debug:
            #generate the lane fit debug image
            debug_fit_lane_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            # draw rectangle
            for rectange_window in rectange_windows:
                win_y_low = rectange_window[0]
                win_y_high = rectange_window[1]
                win_xleft_low = rectange_window[2]
                win_xleft_high = rectange_window[3]
                win_xright_low = rectange_window[4]
                win_xright_high = rectange_window[5]
                cv2.rectangle(debug_fit_lane_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0),
                              2)
                cv2.rectangle(debug_fit_lane_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                              (0, 255, 0), 2)

            # draw the fit points
            debug_fit_lane_img[lefty, leftx] = [255, 0, 0]
            debug_fit_lane_img[righty, rightx] = [0, 0, 255]

            # fit lines, and draw the left and right fit lanes
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])

            cv2.polylines(debug_fit_lane_img, np.int_(left_line_pts), False, (255, 255, 0), 5)
            cv2.polylines(debug_fit_lane_img, np.int_(right_line_pts), False, (255, 255, 0), 5)

            self.debug_manager.lane_fit_image = debug_fit_lane_img
            self.debug_manager.left_fit = left_fit
            self.debug_manager.right_fit = right_fit
        return self.left_fit, self.right_fit

    def tune_lane_line(self, binary_warped):
        """
        * If you have a valid fit from last frame, you could use thif function to avoid doing a blind search again
        * Instead you can just search in a margin around the previous line position

        Parameters
        ----------
        binary_warped: the preprocessed and warped image

        Return
        ----------
        left_fit: the quadratic fit of left lane
        right_fit: the quadratic fit of right lane
        """
        binary_warped[(binary_warped > 0)] = 1
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        left_fit_old = self.left_fit
        right_fit_old = self.right_fit
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit_old[0] * (nonzeroy ** 2) + left_fit_old[1] * nonzeroy + left_fit_old[2] - margin)) &
                          (nonzerox < (left_fit_old[0] * (nonzeroy ** 2) + left_fit_old[1] * nonzeroy + left_fit_old[2] + margin)))

        right_lane_inds = (
                    (nonzerox > (right_fit_old[0] * (nonzeroy ** 2) + right_fit_old[1] * nonzeroy + right_fit_old[2] - margin))
                    & (nonzerox < (right_fit_old[0] * (nonzeroy ** 2) + right_fit_old[1] * nonzeroy + right_fit_old[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
        right_fit_new = np.polyfit(righty, rightx, 2)
        self.left_fit = left_fit_new
        self.right_fit = right_fit_new

        # for debug manager collecting info
        if self.debug:
            #
            debug_fit_lane_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

            # draw the points
            debug_fit_lane_img[lefty, leftx] = [255, 0, 0]
            debug_fit_lane_img[righty, rightx] = [0, 0, 255]

            # fit line of the left and right lane
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            left_fitx = left_fit_new[0] * ploty ** 2 + left_fit_new[1] * ploty + left_fit_new[2]
            right_fitx = right_fit_new[0] * ploty ** 2 + right_fit_new[1] * ploty + right_fit_new[2]

            # draw lanes
            left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])

            cv2.polylines(debug_fit_lane_img, np.int_(left_line_pts), False, (255, 255, 0), 5)
            cv2.polylines(debug_fit_lane_img, np.int_(right_line_pts), False, (255, 255, 0), 5)

            self.debug_manager.lane_fit_image = debug_fit_lane_img
            self.debug_manager.left_fit = left_fit_new
            self.debug_manager.right_fit = right_fit_new
        return self.left_fit, self.right_fit

    def annotate_lane(self, left_fit, right_fit):
        """
        Generate an image which have drawn the lane area

        Parameters
        ----------
        left_fit: the quadratic fit of left lane
        right_fit: the quadratic fit of right lane

        Return
        ----------
        annotated_warped_area: an image which have drawn the lane area in bird's eye view
        """

        # generate the fit points
        ploty = np.linspace(0, self.image_height - 1, self.image_height)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        annotated_warped_area = np.zeros_like(self.input_image)

        # Generate a polygon to illustrate the search window areas
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_contour_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_countour_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_contour_pts = np.hstack((left_contour_pts, right_countour_pts))
        # Draw the lane area
        cv2.fillPoly(annotated_warped_area, np.int_(lane_contour_pts), (0, 255, 0))

        return annotated_warped_area


    def annotate_lane_line(self, left_fit, right_fit):
        """
        Generate an image which have drawn the lane lines

        Parameters
        ----------
        left_fit: the quadratic fit of left lane
        right_fit: the quadratic fit of right lane

        Return
        ----------
        annotated_lines: an image which have drawn the lane lines in perspective view
        """
        ploty = np.linspace(0, self.image_height - 1, self.image_height)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        """
        then we should transform the line points to the perspective view, 
        """
        left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        unwarped_left_line_pts = warp_pts(left_line_pts, self.M_inv)
        unwarped_left_line_pts = unwarped_left_line_pts.reshape(-1, 1, 2)
        unwarped_right_line_pts = warp_pts(right_line_pts, M_inv)
        unwarped_right_line_pts = unwarped_right_line_pts.reshape(-1, 1, 2)
        """
        Finally, we fit the unwarped points to the lane lines and draw the on the picture 
        """
        annotated_lines = np.zeros_like(self.input_image)
        cv2.polylines(annotated_lines, np.int_(unwarped_left_line_pts), True, (255, 0, 0), 10)
        cv2.polylines(annotated_lines, np.int_(unwarped_right_line_pts), True, (255, 0, 0), 10)
        return annotated_lines


    def annotate_road_information(self, image):
        """
        annotate the radius of curvature and vehicle offset on the image

        Parameters
        ----------
        image: input image(np.array())

        Return
        ----------
        image: the image that has lane information
        """

        # the font and color of lane info annotation
        font = cv2.FONT_HERSHEY_COMPLEX
        color = (255, 255, 0)

        road_straight = False
        if self.left_Roc > 0 and self.right_Roc > 0:
            if self.left_Roc > 10000 or self.right_Roc > 10000:
                road_straight = True
            else:
                radius_curvature = (self.left_Roc + self.right_Roc) / 2.0
        elif self.left_Roc < 0 and self.right_Roc < 0:
            if self.left_Roc < -10000 or self.right_Roc < -10000:
                road_straight = True
            else:
                radius_curvature = (self.left_Roc + self.right_Roc) / 2.0
        else:
            road_straight = True
        """
        #2  show the radius of curvature
        """
        if road_straight is True:
            cv2.putText(image, 'Road: nearly straight', (30, 60), font, 1, color, 2)
        elif radius_curvature > 0.0:
            cv2.putText(image, 'Road: RoC is 5.2%f km to the right' % (radius_curvature / 1000),
                        (30, 60), font, 1, color, 2)
        else:
            cv2.putText(image, 'Road: RoC is 5.2%f km to the left' % (- radius_curvature / 1000),
                        (30, 60), font, 1, color, 2)

        """
        #3  get the vehicle offset
        """
        car_position = (self.image_height - 1, self.image_width / 2 - 1)
        vehicle_offset = measure_vehicle_offset(self.left_fit, self.right_fit, car_position)

        """
        #4  show the vehicle offset
        """

        if vehicle_offset < 0.1 and vehicle_offset > -0.1:
            cv2.putText(image, 'Car: in the middle of lane', (30, 90), font, 1, color, 2)
        elif vehicle_offset > 0.1:
            cv2.putText(image, 'Car: %5.2fm right from lane center  ' % vehicle_offset,
                        (30, 90), font, 1, color, 2)
        else:
            cv2.putText(image, 'Car: %5.2fm left from lane center ' % (- vehicle_offset),
                        (30, 90), font, 1, color, 2)

        # for debug manager collecting info
        if self.debug:
            self.debug_manager.left_radius_of_curvature = self.left_Roc
            self.debug_manager.right_radius_of_curvature = self.right_Roc
            self.debug_manager.vehicle_offset = vehicle_offset

        return image


    def adjust_anotated_lane(self):
        """
        Adjust the lane based on the checking result.
            In order to get better results, we made some rules here.
                1. if we failed to pass sanity check in 5 consecutive frames, return False, it means we aren't going to
                            annotate lane on the image
                2. if both left_fit and right_fit valid ,we don't do anything
                3. if left_fit valid but right_fit failed ,we adjust the right_fit based on
                        a.the latest right_fit which passed the sanity check
                        b. the valid left_fit
                4. if left_fit valid but right_fit failed ,we adjust the right_fit based on
                        a.the latest right_fit which passed the sanity check
                        b. the valid left_fit
                5. if neither left_fit or right_fit valid, we adjust two fit based on
                        a.the latest right_fit which passed the sanity check
                2-5 we return True

            we need to add the situation which both fit valid but failed to pass the sanity check

        Parameters
        ----------

        Return
        ----------
        annotate: is the adjusted lane meet the annotation requirement?
        self.adjusted_left_fit: the quadratic fit of adjusted left lane
        self.adjusted_right_fit: the quadratic fit of adjusted right lane
        """
        annotate = True
        if self.con_not_detect_num > 10:
            self.adjusted_right_fit = None
            self.adjusted_left_fit = None
            annotate = False
        elif self.left_fit_valid and self.right_fit_valid:
            self.adjusted_left_fit = self.left_fit
            self.adjusted_right_fit = self.right_fit
        elif self.left_fit_valid and not self.right_fit_valid:
            self.adjusted_left_fit = self.left_fit
            self.adjusted_right_fit = self.predict_right_fit()
        elif self.right_fit_valid and not self.left_fit_valid:
            self.adjusted_right_fit = self.right_fit
            self.adjusted_left_fit = self.predict_left_fit()
        else:
            adjusted_left_fit, adjusted_right_fit = self.predict_both_fits()
            if adjusted_left_fit is None:
                self.adjusted_right_fit = None
                self.adjusted_left_fit = None
                annotate = False
            else:
                self.adjusted_right_fit = adjusted_right_fit
                self.adjusted_left_fit = adjusted_left_fit

        # for debug mananger collecting info
        if self.debug:
            self.debug_manager.annotate = annotate
            self.debug_manager.left_fit_valid = self.left_fit_valid
            self.debug_manager.right_fit_valid = self.right_fit_valid
            self.debug_manager.recent10_check = self.recent_10check
            self.debug_manager.adjusted_left_fit = self.adjusted_left_fit
            self.debug_manager.adjusted_right_fit = self.adjusted_right_fit

            #draw the adjusted lane on the fit_lane_image
            if annotate:
                ploty = np.linspace(0, self.image_height - 1, self.image_height)
                adjusted_left_fitx = self.adjusted_left_fit[0] * ploty ** 2 + self.adjusted_left_fit[1] * ploty + self.adjusted_left_fit[2]
                adjusted_right_fitx = self.adjusted_right_fit[0] * ploty ** 2 + self.adjusted_right_fit[1] * ploty + self.adjusted_right_fit[2]


                left_line_pts = np.array([np.transpose(np.vstack([adjusted_left_fitx, ploty]))])
                right_line_pts = np.array([np.transpose(np.vstack([adjusted_right_fitx, ploty]))])

                cv2.polylines(self.debug_manager.lane_fit_image, np.int_(left_line_pts), False, (0, 255, 255), 3, lineType=4)
                cv2.polylines(self.debug_manager.lane_fit_image, np.int_(right_line_pts), False, (0, 255, 255), 3, lineType=4)

        return annotate, self.adjusted_left_fit, self.adjusted_right_fit

    def sanity_check_video(self):
        """

        sanity check for video
        Parameters
        ----------

        Return
        ----------
        True: we find good fit in recent 5 frames, and we are going to annotate lanes in image
        False: we don't find  any good fit in recent 5 frames,  return false,
                        so we are not going to annotate lane in image
        """
        # check left fit between last frame and current fit
        self.left_fit_valid = self.is_2recent_fit_similar(self.left_fit, self.left_recent_10fitted.get_tail())
        # check right fit between last frame and current fit
        self.right_fit_valid = self.is_2recent_fit_similar(self.right_fit, self.right_recent_10fitted.get_tail())
        # check left and right fit in current frame
        is_parallel = self.is_2lanes_parallel(self.left_fit, self.right_fit)
        has_similar_curvature = self.has_similar_curvature(self.left_fit, self.right_fit)

        # enqueue the fit
        self.left_recent_10fitted.enqueue(self.left_fit)
        self.right_recent_10fitted.enqueue(self.right_fit)

        # if all check are satisfied, we think we had detected the both lanes
        if self.left_fit_valid and self.right_fit_valid and is_parallel and has_similar_curvature:
            self.sanity_check_result = True
            self.con_detect_num += 1
            self.con_not_detect_num = 0
        else:
            self.sanity_check_result = False
            self.con_detect_num = 0
            self.con_not_detect_num += 1
        self.recent_10check.enqueue(self.sanity_check_result)

        # for debug manager collecting info
        if self.debug:
            self.debug_manager.sanity_check_result = self.sanity_check_result
            self.debug_manager.con_detect_num = self.con_detect_num
            self.debug_manager.con_not_detect_num = self.con_not_detect_num

    def sanity_check_image(self):
        """
        sanity check for image

        Parameters
        ----------

        Return
        ----------

        True: we find good fit in recent 5 frames, and we are going to annotate lanes in image
        False: we don't find  any good fit in recent 5 frames,  return false,
                        so we are not going to annotate lane in image
        """
        # check left and right fit in current frame
        is_parallel = self.is_2lanes_parallel(self.left_fit, self.right_fit)
        has_similar_curvature = self.has_similar_curvature(self.left_fit, self.right_fit)

        # if all check are satisfied, we think we had detected the both lanes
        if is_parallel and has_similar_curvature:
            self.sanity_check_result = True
        else:
            self.sanity_check_result = False

        # for debug manager collecting info
        if self.debug:
            self.debug_manager.sanity_check_result = self.sanity_check_result

    def find_lane_line(self, edged_warped):
        """
        try to get reasonable left lane fit and right lane fit

        Parameters
        ----------
        edged_warped: the image has been preprocessed and warped

        Return
        ----------

        the result of adjusted lane
        """

        if self.sanity_check_result == False:
            # if the sanity check of last frame failed, restart to fit the lane in current frame
            self.fit_lane_line(edged_warped)
        else:
            # if the sanity check of last frame success, fit the lane in current frame based on the fit results of last frame
            self.tune_lane_line(edged_warped)

        # check whether the found lane reasonable
        self.sanity_check_video()
        # adjust the lane based on sanity check result and prepare for in coming annotation
        return self.adjust_anotated_lane()

    """
    def detect_ego_lane_line(self, original_img, camera_coeff, M, M_inv):
        #show the ego lane line

        self.input_image = original_img
        self.image_height = original_img.shape[0]
        self.image_width = original_img.shape[1]

        # 1 undistored the image
        mtx = camera_coeff["mtx"]
        dist = camera_coeff["dist"]
        undistored_img = cv2.undistort(original_img, mtx, dist, None, mtx)

        # 2 process the image, get the clear edge information by using color/graident threshold methods(sobel, HLS, and so on...)
        edged_image = image_preprocess2(undistored_img)

        # 3 translate from the edged image from perspective view to bird's eye view
        edged_warped_img = cv2.warpPerspective(edged_image, M, (original_img.shape[1], original_img.shape[0]))

        # 4 fit lines in bird's eye view
        self.fit_lane_line(edged_warped_img)

        # 5 draw lane line based in the fit line in bird's eye view image
        line_image = self.annotate_lane_line(M_inv)

        # 6 Add the lane_image to the original image
        final_image = cv2.addWeighted(original_img, 1, line_image, 0.3, 0)

        # 7 add road information to the image
        final_image = self.annotate_road_information(final_image)

        # for debug manager collecting info
        if self.debug:
            # input image
            self.debug_manager.original_image = original_img
            # edged image
            self.debug_manager.edged_image = edged_image
            # edged and warped image
            self.debug_manager.edged_warped_image = edged_warped_img
            self.debug_manager.frame_num += 1

            final_image = self.debug_manager.debug_info_show1()

        return final_image
        """

    def video_detect(self, original_img, camera_coeff):
        """
        detect and annotate the lane area in video

        Parameters
        ----------
        original_img: the image captured by camera
        camera_coeff: the camera coefficient to undistorted the camera

        Return
        ----------
        final_image: the image has been annotated the lane area
        """

        # the current frame number of the video
        self.frame_num += 1
        # the current frame image
        self.input_image = original_img
        self.image_height = original_img.shape[0]
        self.image_width = original_img.shape[1]

        # 1 undistored the image
        mtx = camera_coeff["mtx"]
        dist = camera_coeff["dist"]
        undistored_img = cv2.undistort(original_img, mtx, dist, None, mtx)

        """
        Before detection begins, we need to initialize some parameters, such as weather condition, surrounding, projection matrix, lane information and so on.
        Getting such information at first can greatly help us to detect correct lanes in following steps. 
        Weather condition: 
        Surrounding:
        Projection matrix: only if we have dst points and corresponding src points, we can obtain a projection matrix. This matrix can help us transfer the perspctive view to bird's eye view(top down view)
        Lane_width, lane_length: we need to estimate the lane width and lane length in pixel for calculating the radius of curvature and sanity check later
        """
        if self.frame_num == 1:
            #self.find_weather_condition()
            #self.find_weather_condition()
            self.get_projection_matrix(undistored_img)

        # 2 process the image, get the clear edge information by using color/graident threshold methods(sobel, HLS, and so on...)
        edged_image = image_preprocess2(undistored_img)

        # 3 translate from the edged image from perspective view to bird's eye view
        edged_warped_img = cv2.warpPerspective(edged_image, self.M, (original_img.shape[1], original_img.shape[0]))
        """
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(original_img)
        ax1.imshow(edged_image, cmap='gray')
        ax2.imshow(edged_warped_img, cmap='gray')
        plt.show()
        """
        # 4 mask for warped image,
        vertices = np.array([[(self.image_width * 0.1, self.image_height), (self.image_width * 0.9, self.image_height),
                              (self.image_width * 0.9, 0), (self.image_width * 0.1, 0)]], dtype=np.int32)

        edged_warped_img = region_of_interest(edged_warped_img, vertices)

        # 5 fit lines in bird's eye view
        annotate, left_fit, right_fit = self.find_lane_line(edged_warped_img)

        # if the found lane line is reasonable, annotate the lane area based on the lane fits
        if annotate:
            # 6 draw lane region based in the fit line in bird's eye view image
            lane_warped_img = self.annotate_lane(left_fit, right_fit)

            # 7 We need to transform back to perspective view(which is same as original image)
            lane_img = cv2.warpPerspective(lane_warped_img, self.M_inv, (original_img.shape[1], original_img.shape[0]))

            # 8 Add the lane_image to the original image
            final_image = cv2.addWeighted(original_img, 1, lane_img, 0.3, 0)

            # 9 add road information to the image
            final_image = self.annotate_road_information(final_image)
        else:
            # if the found lane line is unreasonable, don't annotate
            final_image = original_img

        # for debug manager collecting info
        if self.debug:
            # input image
            self.debug_manager.original_image = original_img
            # edged image
            self.debug_manager.edged_image = np.dstack((edged_image, edged_image, edged_image)) * 255
            # edged and warped image
            self.debug_manager.edged_warped_image = np.dstack((edged_warped_img, edged_warped_img, edged_warped_img)) * 255
            self.debug_manager.frame_num += 1

            final_image = self.debug_manager.debug_video_show()

        return final_image

    def image_detect(self, original_img, camera_coeff):
        """
        detect and annotate the lane area in image

        Parameters
        ----------
        original_img: the image captured by camera
        camera_coeff: the camera coefficient to undistorted the camera

        Return
        ----------
        final_image: the image has been annotated the lane area
        """
        self.input_image = original_img
        self.image_height = original_img.shape[0]
        self.image_width = original_img.shape[1]

        # 1 undistored the image
        mtx = camera_coeff["mtx"]
        dist = camera_coeff["dist"]
        undistored_img = cv2.undistort(original_img, mtx, dist, None, mtx)

        """
        Before detection begins, we need to initialize some parameters, such as weather condition, surrounding, projection matrix, lane information and so on.
        Getting such information at first can greatly help us to detect correct lanes in following steps. 
        Weather condition: 
        Surrounding:
        Projection matrix: only if we have dst points and corresponding src points, we can obtain a projection matrix. This matrix can help us transfer the perspctive view to bird's eye view(top down view)
        Lane_width, lane_length: we need to estimate the lane width and lane length in pixel for calculating the radius of curvature and sanity check later
        """

        #self.find_weather_condition()
        #self.find_weather_condition()
        self.get_projection_matrix(undistored_img)

        # 2 process the image, get the clear edge information by using color/graident threshold methods(sobel, HLS, and so on...)
        edged_image = image_preprocess2(undistored_img)

        # 3 translate from the edged image from perspective view to bird's eye view
        edged_warped_img = cv2.warpPerspective(edged_image, self.M, (original_img.shape[1], original_img.shape[0]))

        """
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(original_img)
        ax1.imshow(edged_image, cmap='gray')
        ax2.imshow(edged_warped_img, cmap='gray')
        plt.show()
        """

        #4 mask for warped image,
        vertices = np.array([[(self.image_width * 0.1, self.image_height), (self.image_width * 0.9, self.image_height),
                              (self.image_width * 0.9, 0), (self.image_width * 0.1, 0)]], dtype=np.int32)

        edged_warped_img = region_of_interest(edged_warped_img, vertices)

        # 5 fit lines in bird's eye view
        left_fit, right_fit = self.fit_lane_line(edged_warped_img)
        self.sanity_check_image()

        # 6 draw lane region based in the fit line in bird's eye view image
        lane_warped_img = self.annotate_lane(left_fit, right_fit)

        # 7 We need to transform back to perspective view(which is same as original image)
        lane_img = cv2.warpPerspective(lane_warped_img, self.M_inv, (original_img.shape[1], original_img.shape[0]))

        # 8 Add the lane_image to the original image
        final_image = cv2.addWeighted(original_img, 1, lane_img, 0.3, 0)

        # 9 add road information to the image
        final_image = self.annotate_road_information(final_image)

        # for debug manager collecting info
        if self.debug:
            # input image
            self.debug_manager.original_image = original_img
            # edged image
            self.debug_manager.edged_image = np.dstack((edged_image, edged_image, edged_image)) * 255
            # edged and warped image
            self.debug_manager.edged_warped_image = np.dstack((edged_warped_img, edged_warped_img, edged_warped_img)) * 255
            self.debug_manager.lane_annotate_image = final_image
            self.debug_manager.frame_num += 1

            final_image = self.debug_manager.debug_image_show()

        return final_image


if __name__ == "__main__":

    img = mpimg.imread('test_images/test6.jpg')

    calibrate_camera()
    # get the coefficients of camera
    camera_coeff = pickle.load(open("camera_cal/camera_coeff.p", "rb"))
    road_lane = Lanes(debug=True)
    road_lane.image_detect(img, camera_coeff)

    plt.show()
