import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from final_project.Advanced_Lane_Lines.my_datastruct import *
from final_project.Advanced_Lane_Lines.debug_manager import *
from final_project.Advanced_Lane_Lines.transform_perspective import *
from final_project.Advanced_Lane_Lines.extract_lane_info import *
from final_project.Advanced_Lane_Lines.calibrate_camera import *


class Lanes(object):
    def __init__(self, debug=False):
        self.frame_num = 0
        self.input_image = None
#        self.final_image = None
        self.image_height = None
        self.image_width = None

        """
        if lane line in 5 frames is consecutively detected, we can judge the lines are detected.
        if lane line in 5 frames is consecutively failed to be detected, we can judge the lines are detected.    
        """
        self.sanity_check_result = False
        self.recent_10check = MyCirculateQueue(10)
        # the frame number that detect lane line consecutively
        self.con_detect_num = 0
        # the frame number that do not detect lane line consecutively
        self.con_not_detect_num = 0

        # annotate the lane in the area?
        self.annotate = False
        # self.current_fit = [np.array([False])]
        self.left_fit = np.array([False])                #[np.array([False])]
        self.right_fit = np.array([False])               #[np.array([False])]
        self.adjusted_left_fit = np.array([False])       #[np.array([False])]
        self.adjusted_right_fit = np.array([False])      #[np.array([False])]
        # is fit reasonable?
        self.left_fit_valid = False
        self.right_fit_valid = False
        # the last n fits of the line
        self.left_recent_10fitted = MyCirculateQueue(10)
        self.right_recent_10fitted = MyCirculateQueue(10)

        # for projection from perspective view to bird's eye view
        self.perspective_src_points = None
        self.perspective_dst_points = np.float32(([384, 720], [896, 720], [896, 600], [384, 600]))
        self.M = None
        self.M_inv = None

        # lane_width and lane_length in pixel
        self.lane_width = 896-384
        self.lane_length = None

        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit


        #radius of curvature of the line in some units
        #suppose the initial lane is straight
        self.right_Roc = 10000.
        self.left_Roc = 10000.

        #distance in meters of vehicle center from the lane center
        self.vehicle_offset = 0.
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.debug = debug
        if self.debug is True:
            self.debug_manager = DebugManager()

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
        The Roc calculated value varies, so we think most situations are reasonable, except:
        1.  0 < left_Roc < 2000 and -2000 < left_Roc < 0
        2.  0 < right_Roc < 2000 and -2000 < right_Roc < 0
        :return: True or False
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
        :return:
        """

        check_result = True
        ploty = np.linspace(0, self.image_height - 1, self.image_height / 10)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        distances = right_fitx - left_fitx


        diffs = distances - self.lane_width
        threshold = 250
        for diff in diffs:
            if diff > threshold or diff < -threshold:
                check_result = False

        mse = np.sqrt(np.sum(np.square(diffs)) / len(diffs))
        if mse > threshold:
            check_result = False

        if self.debug:
            self.debug_manager.parallel_mse = mse

        return check_result

    def is_2recent_fit_similar(self, current_fit, last_fit):
        """
        Check whether the lane roughly equal between two frames
        :return: True or False
        """

        ret = True
        # for the first frame, the last_fit = None
        if last_fit is None:
            last_fit = np.array([0, 0, 0])

        ploty = np.linspace(0, self.image_height - 1, self.image_height / 10)
        current_fitx = current_fit[0] * ploty ** 2 + current_fit[1] * ploty + current_fit[2]
        last_fitx = last_fit[0] * ploty ** 2 + last_fit[1] * ploty + last_fit[2]
        diffs = last_fitx - current_fitx
        threshold = 70

        for diff in diffs:
            if diff > threshold or diff < - threshold:
                ret = False

        #mean square error
        mse = np.sqrt(np.sum(np.square(diffs)) / len(diffs))
        if mse > threshold / 2:
            ret = False

        if self.debug:
            if self.left_fit is current_fit:
                self.debug_manager.left_fit_mse = mse
            elif self.right_fit is current_fit:
                self.debug_manager.right_fit_mse = mse

        return ret


    def get_latest_valid_fit(self, which_lane):
        assert which_lane == 'left' or which_lane == 'right'
        index = self.recent_10check.find(True)
        if index is None:
            return None
        if which_lane == 'left':
            valid_fit = self.left_recent_10fitted.visit(index)
        elif which_lane == 'right':
            valid_fit = self.right_recent_10fitted.visit(index)
        return valid_fit

    def predict_right_fit(self):
        latest_valid_right_fit = self.get_latest_valid_fit('right')
        if (latest_valid_right_fit is not None) and self.is_2lanes_parallel(self.left_fit, latest_valid_right_fit) and \
                self.has_similar_curvature(self.left_fit, latest_valid_right_fit):
            adjusted_right_fit = latest_valid_right_fit
        else:
            left_fit = self.left_fit
            left_fit[2] += self.lane_width
            adjusted_right_fit = left_fit
        return adjusted_right_fit

    def predict_left_fit(self):
        latest_valid_left_fit = self.get_latest_valid_fit('left')
        if (latest_valid_left_fit is not None) and self.is_2lanes_parallel(latest_valid_left_fit, self.right_fit) and \
                self.has_similar_curvature(latest_valid_left_fit, self.right_fit):
            adjusted_left_fit = latest_valid_left_fit
        else:
            right_fit = self.right_fit
            right_fit[2] -= self.lane_width
            adjusted_left_fit = right_fit
        return adjusted_left_fit

    def predict_both_fits(self):
        latest_valid_right_fit = self.get_latest_valid_fit('right')
        latest_valid_left_fit = self.get_latest_valid_fit('left')
        return latest_valid_left_fit, latest_valid_right_fit

    def get_projection_matrix(self, undistored_img):
        #  process the image, get the clear edge information by using color/graident threshold methods)
        edged_image = extract_lane_information3(undistored_img)

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


        if src_points is not None:
            self.perspective_src_points = src_points

            # generate the projection matrix
            M, M_inv = transform_perspective(self.perspective_src_points, self.perspective_dst_points)
            self.M = M
            self.M_inv = M_inv


    def fit_lane_line(self, binary_warped):
        """finding lane through sliding windows"""
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

            #only for debug
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
            #
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

            # draw the points
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
            plt.imshow(debug_fit_lane_img)
            plt.show()
            self.debug_manager.lane_fit_image = debug_fit_lane_img
            self.debug_manager.left_fit = left_fit
            self.debug_manager.right_fit = right_fit

    def tune_lane_line(self, binary_warped):
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

    def annotate_lane(self):

        ploty = np.linspace(0, self.image_height - 1, self.image_height)
        left_fitx = self.adjusted_left_fit[0] * ploty ** 2 + self.adjusted_left_fit[1] * ploty + self.adjusted_left_fit[2]
        right_fitx = self.adjusted_right_fit[0] * ploty ** 2 + self.adjusted_right_fit[1] * ploty + self.adjusted_right_fit[2]

        annotated_warped_area = np.zeros_like(self.input_image)

        # Generate a polygon to illustrate the search window areas
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_contour_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_countour_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_contour_pts = np.hstack((left_contour_pts, right_countour_pts))
        # Draw the road onto the original image
        cv2.fillPoly(annotated_warped_area, np.int_(lane_contour_pts), (0, 255, 0))

        return annotated_warped_area


    def annotate_lane_line(self, M_inv):
        """
           We firstly get the line points in the bird's eye view,
           """
        ploty = np.linspace(0, self.image_height - 1, self.image_height)
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        """
        then we should transform the line points to the perspective view, 
        """
        left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        unwarped_left_line_pts = warp_pts(left_line_pts, M_inv)
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

    def annotate_road_information(self, final_image):
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
            cv2.putText(final_image, 'Road: nearly straight', (30, 60), font, 1, color, 2)
        elif radius_curvature > 0.0:
            cv2.putText(final_image, 'Road: RoC is 5.2%f km to the right' % (radius_curvature / 1000),
                        (30, 60), font, 1, color, 2)
        else:
            cv2.putText(final_image, 'Road: RoC is 5.2%f km to the left' % (- radius_curvature / 1000),
                        (30, 60), font, 1, color, 2)

        """
        #3  get the vehicle offset
        """
        car_position = (self.image_height -1, self.image_width / 2 - 1)
        vehicle_offset = measure_vehicle_offset(self.left_fit, self.right_fit, car_position)

        """
        #4  show the vehicle offset
        """

        if vehicle_offset < 0.1 and vehicle_offset > -0.1:
            cv2.putText(final_image, 'Car: in the middle of lane', (30, 90), font, 1, color, 2)
        elif vehicle_offset > 0.1:
            cv2.putText(final_image, 'Car: %5.2fm right from lane center  ' % vehicle_offset,
                        (30, 90), font, 1, color, 2)
        else:
            cv2.putText(final_image, 'Car: %5.2fm left from lane center ' % (- vehicle_offset),
                        (30, 90), font, 1, color, 2)

        # for debug manager collecting info
        if self.debug:
            self.debug_manager.left_radius_of_curvature = self.left_Roc
            self.debug_manager.right_radius_of_curvature = self.right_Roc
            self.debug_manager.vehicle_offset = vehicle_offset
            self.debug_manager.lane_annotate_image = final_image

        return final_image


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


        return annotate

    def sanity_check(self):
        """
        sanity check for video
        :return:  True: we find good fit in recent 5 frames, and we are going to annotate lanes in image
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

        if self.debug:
            self.debug_manager.sanity_check_result = self.sanity_check_result
            self.debug_manager.con_detect_num = self.con_detect_num
            self.debug_manager.con_not_detect_num = self.con_not_detect_num

    def find_lane_line(self, edged_warped):

        self.image_height = np.int(edged_warped.shape[0])
        self.image_width = np.int(edged_warped.shape[1])

        if self.sanity_check_result == False:
            self.fit_lane_line(edged_warped)
        else:
            self.tune_lane_line(edged_warped)

        # check whether the found lane reasonable
        self.sanity_check()
        # adjust the lane based on sanity check result and prepare for in coming annotation
        self.annotate = self.adjust_anotated_lane()
        return self.annotate

    def detect_ego_lane_line(self, original_img, camera_coeff, M, M_inv):
        """show the ego lane line"""

        self.input_image = original_img
        self.image_height = original_img.shape[0]
        self.image_width = original_img.shape[1]

        # 1 undistored the image
        mtx = camera_coeff["mtx"]
        dist = camera_coeff["dist"]
        undistored_img = cv2.undistort(original_img, mtx, dist, None, mtx)

        # 2 process the image, get the clear edge information by using color/graident threshold methods(sobel, HLS, and so on...)
        edged_image = extract_lane_information2(undistored_img)

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

    def detect_lane(self, original_img, camera_coeff):
        self.frame_num += 1
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
        edged_image = extract_lane_information2(undistored_img)

        # 3 translate from the edged image from perspective view to bird's eye view
        edged_warped_img = cv2.warpPerspective(edged_image, self.M, (original_img.shape[1], original_img.shape[0]))
        """
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(original_img)
        ax1.imshow(edged_image, cmap='gray')
        ax2.imshow(edged_warped_img, cmap='gray')
        plt.show()
        """
        vertices = np.array([[(128, 720), (1280 - 128, 720), (1280 -128, 0), (128, 0)]], dtype=np.int32)
        edged_warped_img = region_of_interest(edged_warped_img, vertices)

        # 4 fit lines in bird's eye view
        annotate = self.find_lane_line(edged_warped_img)

        if annotate:
            # 5 draw lane region based in the fit line in bird's eye view image
            lane_warped_img = self.annotate_lane()

            # 6 We need to transform back to perspective view(which is same as original image)
            lane_img = cv2.warpPerspective(lane_warped_img, self.M_inv, (original_img.shape[1], original_img.shape[0]))

            # 7 Add the lane_image to the original image
            final_image = cv2.addWeighted(original_img, 1, lane_img, 0.3, 0)

            # 8 add road information to the image
            final_image = self.annotate_road_information(final_image)
        else:
            # don't annotate the image
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

            final_image = self.debug_manager.debug_info_show1()

        return final_image


if __name__ == "__main__":

    img = mpimg.imread('test_images/test6.jpg')

    calibrate_camera()
    # get the coefficients of camera
    camera_coeff = pickle.load(open("camera_cal/camera_coeff.p", "rb"))
    road_lane = Lanes(debug=True)
    road_lane.detect_lane(img, camera_coeff)

    plt.show()
