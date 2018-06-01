import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from transform_perspective import *


def make_half(image, half='bottom'):
    """
        Define a function to chop a picture in half horizontally

        Parameters
        ----------
        image: the long video need to be clipped
        half: 'bottom' or 'top'

        Return
        ----------
        the half of the image

        """

    assert half == 'top' or half == 'bottom'

    image_height = image.shape[0]
    if half == 'bottom':
        # get the bottom half of the image
        if len(image.shape) < 3:
            newimage = np.copy(image[image_height / 2:image_height, :])
        else:
            newimage = np.copy(image[image_height / 2:image_height, :, :])
    else:
        # get the top half of image
        if len(image.shape) < 3:
            newimage = np.copy(image[0:image_height / 2, :])
        else:
            newimage = np.copy(image[0:image_height / 2, :, :])
    return newimage


def image_quality(image):
    """
        Define a function to check image quality

        Parameters
        ----------
        image: the image to be checked (It is better to be the undistorted image)

        Return
        ----------
        sky_image_quality: the image quality of sky
        sky_text: the condition of sky
        road_image_quality: the image quality of road

        """
    image_height = image.shape[0]
    yuv_img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV).astype(np.float32)

    # get some stats for the sky image
    sky_lightness_img = yuv_img[0:image_height//2, :, 0]  # the Y of the top half image
    sky_rgb_img = image[0:image_height//2, :, :]
    sky_lightness = np.average(sky_lightness_img)
    sky_red = np.average(sky_rgb_img[:, :, 0])
    sky_green = np.average(sky_rgb_img[:, :, 1])
    sky_blue = np.average(sky_rgb_img[:, :, 2])

    # Sky image condition
    if sky_lightness > 160:
        sky_image_quality = 'Sky Image: overexposed'
    elif sky_lightness < 50:
        sky_image_quality = 'Sky Image: underexposed'
    elif sky_lightness > 143:
        sky_image_quality = 'Sky Image: normal bright'
    elif sky_lightness < 113:
        sky_image_quality = 'Sky Image: normal dark'
    else:
        sky_image_quality = 'Sky Image: normal'

    # Sky detected weather or lighting conditions
    if sky_lightness > 128:
        if sky_blue > sky_lightness:
            if sky_red > 120 and sky_green > 120:
                if (sky_green - sky_red) > 20.0:
                    sky_text = 'Sky Condition: tree shaded'
                else:
                    sky_text = 'Sky Condition: cloudy'
            else:
                sky_text = 'Sky Condition: clear'
        else:
            sky_text = 'Sky Condition: UNKNOWN SKYL > 128'
    else:
        if sky_green > sky_blue:
            sky_text = 'Sky Condition: surrounded by trees'
            #visibility = -80
        elif sky_blue > sky_lightness:
            if (sky_green - sky_red) > 10.0:
                sky_text = 'Sky Condition: tree shaded'
            else:
                sky_text = 'Sky Condition: very cloudy or under overpass'
        else:
            sky_text = 'Sky Condition: UNKNOWN!'

    # get some stats for the sky image
    road_lightness_img = yuv_img[image_height // 2:, :, 0]  # the Y of the top half image
    road_rgb_img = image[image_height // 2:, :, :]
    road_lightness = np.average(road_lightness_img)
    road_red = np.average(road_rgb_img[:, :, 0])
    road_green = np.average(road_rgb_img[:, :, 1])
    road_blue = np.average(road_rgb_img[:, :, 2])

    #roadbalance = road_lightness / 10.0

    # Road image condition
    if road_lightness > 160:
        road_image_quality = 'Road Image: overexposed'
    elif road_lightness < 50:
        road_image_quality = 'Road Image: underexposed'
    elif road_lightness > 143:
        road_image_quality = 'Road Image: normal bright'
    elif road_lightness < 113:
        road_image_quality = 'Road Image: normal dark'
    else:
        road_image_quality = 'Road Image: normal'

    return sky_image_quality, sky_text, road_image_quality


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    """
    Define a function that takes an image, gradient orientation, and threshold min / max values.

    Parameters
    ----------
    img : the image need to be processed
    orient: 'x' sobelx, apply the graident in x direction, 'y' sobely, apply the graident in y direction
    thresh(min_ threshold, max_threshold):  the minimum and maximum value of threshold,
                                minimum must be >= 0, maximum must be <= 255

    Return
    ----------
    the thresholded image

    """
    assert orient == 'x' or orient == 'y'
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Define a function to return the magnitude of the gradient for a given sobel kernel size and threshold values

    Parameters
    ----------
    img : the image need to be processed
    sobel_kernel: soble kernel size, which must be positive and odd
        thresh(min_ threshold, max_threshold):  the minimum and maximum value of threshold,
                                minimum must be >= 0, maximum must be <= 255

    Return
    ----------
    the thresholded image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Define a function that applies Sobel x and y, then computes the direction of the gradient, and applies a threshold.

    Parameters
    ----------
    img : the image need to be processed
    sobel_kernel: soble kernel size, which must be positive and odd
    thresh(min_ threshold, max_threshold):  the minimum and maximum value of threshold,
                                minimum must be >= 0, maximum must be <= 255

    Return
    ----------
    the thresholded image
    """

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    # 6) Return this mask as your binary_output image
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def hls_threshold(img, h_threshold=(0, 255), l_threshold=(0, 255), s_threshold=(0, 255)):
    """
    Define a function that applies HLS threshold.

    Parameters
    ----------
    img : the image need to be processed
    h_threshold: the range of threshold for Hue
    l_threshold: the range of threshold for Lightness
    s_threshold: the range of threshold for Saturation

    Return
    ----------
    the thresholded image
    """

    # tranform the RGB/BGR color space to HLS color space
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # get H/L/S channel of image
    h_img = hls_img[:, :, 0]
    l_img = hls_img[:, :, 1]
    s_img = hls_img[:, :, 2]
    binary_output = np.zeros_like(h_img)
    # apply the color threshold to the binary image
    binary_output[(h_img >= h_threshold[0]) & (h_img <= h_threshold[1]) & (l_img >= l_threshold[0]) &
                  (l_img <= l_threshold[1]) & (s_img >= s_threshold[0]) & (s_img <= s_threshold[1])] = 1
    return binary_output


def hsv_threshold(img, h_threshold=(0, 255), s_threshold=(0, 255), v_threshold=(0, 255)):
    """
    Define a function that applies HSV threshold.

    Parameters
    ----------
    img : the image need to be processed
    h_threshold: the range of threshold for Hue
    s_threshold: the range of threshold for Saturation
    v_threshold: the range of threshold for Value

    Return
    ----------
    the thresholded image
    """
    #tranform the RGB/BGR color space to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ## get H/S/V channel of image
    h_img = hsv_img[:, :, 0]
    s_img = hsv_img[:, :, 1]
    v_img = hsv_img[:, :, 2]
    binary_output = np.zeros_like(h_img)
    #apply the color threshold to the binary image
    binary_output[(h_img >= h_threshold[0]) & (h_img <= h_threshold[1]) & (v_img >= v_threshold[0]) &
                  (v_img <= v_threshold[1]) & (s_img >= s_threshold[0]) & (s_img <= s_threshold[1])] = 1
    return binary_output


def yuv_threshold(img, y_threshold=(0, 255), u_threshold=(0, 255), v_threshold=(0, 255)):
    """
    Define a function that applies HSV threshold.

    Parameters
    ----------
    img : the image need to be processed
    h_threshold: the range of threshold for Hue
    s_threshold: the range of threshold for Saturation
    v_threshold: the range of threshold for Value

    Return
    ----------
    the thresholded image
    """
    #tranform the RGB/BGR color space to YUV color space
    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # get Y/U/V channel of image
    y_img = yuv_img[:, :, 0]
    u_img = yuv_img[:, :, 1]
    v_img = yuv_img[:, :, 2]
    binary_output = np.zeros_like(y_img)
    #apply the color threshold to the binary image
    binary_output[(y_img >= y_threshold[0]) & (y_img <= y_threshold[1]) & (u_img >= u_threshold[0]) &
                  (u_img <= u_threshold[1]) & (v_img >= v_threshold[0]) & (v_img <= v_threshold[1])] = 1
    return binary_output


def rgb_threshold(img, r_threshold=(0, 255), g_threshold=(0, 255), b_threshold=(0, 255)):
    """
    Define a function that applies HSV threshold.

    Parameters
    ----------
    img : the image need to be processed
    r_threshold: the range of threshold for Hue
    g_threshold: the range of threshold for Saturation
    b_threshold: the range of threshold for Value

    Return
    ----------
    the thresholded image
    """
    #do not need to tranform to RGB color space, because now is RGB color space
    #rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGB)
    # get R/G/B channel of image
    r_img = img[:, :, 0]
    g_img = img[:, :, 1]
    b_img = img[:, :, 2]
    binary_output = np.zeros_like(r_img)
    #apply the color threshold to the binary image
    binary_output[(r_img >= r_threshold[0]) & (r_img <= r_threshold[1]) & (g_img >= g_threshold[0]) &
                  (g_img <= g_threshold[1]) & (b_img >= b_threshold[0]) & (b_img <= b_threshold[1])] = 1
    return binary_output


def region_of_interest(img, vertices):
    """
    ROI (region of interest )
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    Parameters
    ----------
    img : the image need to be processed
    vertices: the points that make up the vertices

    Return
    ----------
    the ROI image
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def filter_bad_lines(lines, threshold):
    """
    Define a function that filtrate out some disturbing lines

    Parameters
    ----------
    lines : the lines get from HoughLinesP
    threshold: the diff value to filter the bad line

    Return
    ----------
    the lines have same slope
    """
    if lines is None:
        return None
    # compute slope
    slope = [(x2 - x1) / (y2 - y1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        # compute the mean of slope
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        # find the worst line
        idx = np.argmax(diff)
        # if the worst line is bad, remove it.
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            return lines


def estimate_src_points(img_shape, lines):
    """
    Define a function that estimate the source points of perspective transform,
    so that we can calculate the projection matrix

    Parameters
    ----------
    img_shape : the shape of image
    lines: the straight line

    Return
    ----------
    the source points
    """
    height = img_shape[0]
    width = img_shape[1]
    # seperate the lines into left group and right group,
    left_lines = []
    right_lines = []
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # fit the straight line and compute slope and intercept
                fit = np.polyfit((y1, y2), (x1, x2), 1)
                slope = fit[0]
                intercept = fit[1]
                # the x value of intersection of fit line and image bottom
                cross_x = slope * height + intercept

                if -2 < slope < -0.4 and -width * 0.2 <= cross_x <= width / 2:
                    left_lines.append(line)
                elif 0.4 < slope < 2 and width / 2 <= cross_x <= width * 1.2:
                    right_lines.append(line)

        # filtrate the distrubing lines
        filter_bad_lines(left_lines, 0.1)
        filter_bad_lines(right_lines, 0.1)

        if left_lines and right_lines:
            leftx = []
            lefty = []
            rightx = []
            righty = []

            leftx.extend([x1 for line in left_lines for x1, y1, x2, y2 in line])
            leftx.extend([x2 for line in left_lines for x1, y1, x2, y2 in line])
            lefty.extend([y1 for line in left_lines for x1, y1, x2, y2 in line])
            lefty.extend([y2 for line in left_lines for x1, y1, x2, y2 in line])

            rightx.extend([x1 for line in right_lines for x1, y1, x2, y2 in line])
            rightx.extend([x2 for line in right_lines for x1, y1, x2, y2 in line])
            righty.extend([y1 for line in right_lines for x1, y1, x2, y2 in line])
            righty.extend([y2 for line in right_lines for x1, y1, x2, y2 in line])

            # fit left straight line
            left_straight_fit = np.polyfit(lefty, leftx, 1)
            # fit right straight line
            right_straight_fit = np.polyfit(righty, rightx, 1)

            k1 = left_straight_fit[0]
            b1 = left_straight_fit[1]
            k2 = right_straight_fit[0]
            b2 = right_straight_fit[1]

            """
             find the right and left line's intercept, which means solve the following two equations
             x = k1 * y + b1
             x = k2 * y + b2
             solve for (x, y): the intercept of the left and right lines
             which is:  x = (k1 * b2 - k2 * b1) / (k1 - k2)
             and        y = (b2 -b1) / (k1 - k2)
            """

            # intersect_x = int((k1 * b2 - k2 * b1) / (k1 - k2))
            # intersect_y = int((b2 -b1) / (k1 - k2))

            # get four src points for projection
            # generate src rect for projection of road to flat plane
            bottom_y = height
            #top_y = height - (height - intersect_y) * 0.8
            top_y = height * 0.8

            #left_bottom
            p1 = [int(k1 * bottom_y + b1), int(bottom_y)]
            #right_bottom
            p2 = [int(k2 * bottom_y + b2), int(bottom_y)]
            #right_top
            p3 = [int(k2 * top_y + b2), int(top_y)]
            #left_top
            p4 = [int(k1 * top_y + b1), int(top_y)]

            src_points = np.float32((p1, p2, p3, p4))

            return src_points
        else:
            return None
    except:
        return None


def estimate_src_points1(img_shape, lines):
    """
    backup one
    Define function that estimate the source points of perspective transform,
    so that we can calculate the projection matrix

    Parameters
    ----------
    img_shape : the shape of image
    lines: the straight line

    Return
    ----------
    the source points
    """
    backoff = 30
    ysize = img_shape[0]
    midleft = img_shape[1] / 2 - 200 + backoff * 2
    midright = img_shape[1] / 2 + 200 - backoff * 2
    top = ysize / 2 + backoff * 2
    rightslopemin = 0.5  # 8/backoff
    rightslopemax = 3.0  # backoff/30
    leftslopemax = -0.5  # -8/backoff
    leftslopemin = -3.0  # -backoff/30
    try:
        # rightline and leftline cumlators
        rl = {'num': 0, 'slope': 0.0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        ll = {'num': 0, 'slope': 0.0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = ((y2 - y1) / (x2 - x1))
                sides = (x1 + x2) / 2
                vmid = (y1 + y2) / 2
                if slope > rightslopemin and slope < rightslopemax and sides > midright and vmid > top:  # right

                    rl['num'] += 1
                    rl['slope'] += slope
                    rl['x1'] += x1
                    rl['y1'] += y1
                    rl['x2'] += x2
                    rl['y2'] += y2
                elif slope > leftslopemin and slope < leftslopemax and sides < midleft and vmid > top:  # left
                    ll['num'] += 1
                    ll['slope'] += slope
                    ll['x1'] += x1
                    ll['y1'] += y1
                    ll['x2'] += x2
                    ll['y2'] += y2

        if rl['num'] > 0 and ll['num'] > 0:
            # average/extrapolate all of the lines that makes the right line
            rslope = rl['slope'] / rl['num']
            rx1 = int(rl['x1'] / rl['num'])
            ry1 = int(rl['y1'] / rl['num'])
            rx2 = int(rl['x2'] / rl['num'])
            ry2 = int(rl['y2'] / rl['num'])

            # average/extrapolate all of the lines that makes the left line
            lslope = ll['slope'] / ll['num']
            lx1 = int(ll['x1'] / ll['num'])
            ly1 = int(ll['y1'] / ll['num'])
            lx2 = int(ll['x2'] / ll['num'])
            ly2 = int(ll['y2'] / ll['num'])

            # find the right and left line's intercept, which means solve the following two equations
            # rslope = ( yi - ry1 )/( xi - rx1)
            # lslope = ( yi = ly1 )/( xi - lx1)
            # solve for (xi, yi): the intercept of the left and right lines
            # which is:  xi = (ly2 - ry2 + rslope*rx2 - lslope*lx2)/(rslope-lslope)
            # and        yi = ry2 + rslope*(xi-rx2)
            xi = int((ly2 - ry2 + rslope * rx2 - lslope * lx2) / (rslope - lslope))
            yi = int(ry2 + rslope * (xi - rx2))

            # calculate backoff from intercept for right line
            if rslope > rightslopemin and rslope < rightslopemax:  # right
                #ry1 = yi + int(backoff)
                ry1 = ysize - 0.3 *(ysize - yi)
                rx1 = int(rx2 - (ry2 - ry1) / rslope)
                ry2 = ysize - 1
                rx2 = int(rx1 + (ry2 - ry1) / rslope)

            # calculate backoff from intercept for left line
            if lslope < leftslopemax and lslope > leftslopemin:  # left
                #ly1 = yi + int(backoff)
                ly1 = ysize - 0.3 * (ysize - yi)
                lx1 = int(lx2 - (ly2 - ly1) / lslope)
                ly2 = ysize - 1
                lx2 = int(lx1 + (ly2 - ly1) / lslope)

        src_points = np.float32(((lx2, ly2), (rx2, ry2), (rx1, ry1), (lx1, ly1)))

        # return the left and right line slope, found rectangler box shape and the estimated vanishing point.
        return src_points
    except:
        return None



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    """
    generate a set of hough lines and get the source

    Parameters
    ----------
    img: should be the output of a Canny-like transform.
    lines: the straight line

    Return
    ----------
    the source points
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    src_points = estimate_src_points((img.shape[0], img.shape[1]), lines)

    return src_points



def hough_lines1(masked_edges):
    """
    hough line version 1

    Parameters
    ----------
    img: should be the output of a Canny-like transform.

    Return
    ----------
    the source points
    """
    # Define the Hough transform parameters
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 120  # 50 75 25 minimum number of pixels making up a line
    max_line_gap = 40  # 40 50 20 maximum gap in pixels between connectable line segments
    return hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)


def hough_lines2(masked_edges):
    """
        hough line version 2

        Parameters
        ----------
        img: should be the output of a Canny-like transform.

        Return
        ----------
        the source points
        """

    # Define the Hough transform parameters
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 75  # 50 75 25 minimum number of pixels making up a line
    max_line_gap = 40  # 40 50 20 maximum gap in pixels between connectable line segments
    return hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)


def hough_lines3(masked_edges):
    """
        hough line version 3

        Parameters
        ----------
        img: should be the output of a Canny-like transform.

        Return
        ----------
        the source points
        """
    # Define the Hough transform parameters
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25  # 50 75 25 minimum number of pixels making up a line
    max_line_gap = 20  # 40 50 20 maximum gap in pixels between connectable line segments
    return hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)


def hough_lines4(masked_edges):
    """
        hough line version 4

        Parameters
        ----------
        img: should be the output of a Canny-like transform.

        Return
        ----------
        the source points
        """
    # Define the Hough transform parameters
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 40  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # 50 75 25 minimum number of pixels making up a line
    max_line_gap = 20  # 40 50 20 maximum gap in pixels between connectable line segments
    return hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)


def image_preprocess1(img):
    """
        image preprocess version 1
        using: yellow threshold, white threshold, ROI

        Parameters
        ----------
        img: image (np.array())

        Return
        ----------
        the source points
    """

    # set white and yellow threshold
    white = rgb_threshold(img, r_threshold=(200, 255), g_threshold=(200, 255), b_threshold=(200, 255))
    yellow = hsv_threshold(img, h_threshold=(20, 34), s_threshold=(43, 255), v_threshold=(46, 255))

    # filtrate the image using only color information (white lane -- RGB space) (yellow lane -- HSV space)
    combined = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    combined[(white == 1) | (yellow == 1)] = 1

    # region of interest
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array([[(0, height), (width, height), (width * 0.6, height * 0.5), (width * 0.4, height * 0.5)]], dtype=np.int32)
    roi_image = region_of_interest(img, vertices)

    return roi_image


def image_preprocess2(img):
    """
        image preprocess version 2
        using: yellow threshold, white threshold, sobelX, sobelY, ROI

        Parameters
        ----------
        img: image (np.array())

        Return
        ----------
        the source points
    """

    # set white and yellow threshold
    white = rgb_threshold(img, r_threshold=(200, 255), g_threshold=(200, 255), b_threshold=(200, 255))
    yellow = hsv_threshold(img, h_threshold=(20, 34), s_threshold=(43, 255), v_threshold=(46, 255))
    # set sobelX and sobelY threshold
    gradx = abs_sobel_thresh(img, orient='x', thresh=(35, 120))
    grady = abs_sobel_thresh(img, orient='y', thresh=(30, 120))

    # filtrate the image using color information (white lane -- RGB space) (yellow lane -- HSV space)
    # and graident information (SobleX, Soble Y)
    combined = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    combined[(white == 1) | (yellow == 1) | ((gradx == 1) & (grady == 1))] = 1

    # region of interest
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array([[(0, height), (width, height), (width * 0.6, height * 0.5), (width * 0.4, height * 0.5)]],
                        dtype=np.int32)
    roi_image = region_of_interest(img, vertices)
    return roi_image


def image_preprocess3(img):

    """
        image preprocess version 3
        using: yellow threshold, white threshold, sobelX, sobelY, magnitude gradient, direction gradient  ROI

        Parameters
        ----------
        img: image (np.array())

        Return
        ----------
        the source points
    """

    # set white and yellow threshold
    white = rgb_threshold(img, r_threshold=(200, 255), g_threshold=(200, 255), b_threshold=(200, 255))
    yellow = hsv_threshold(img, h_threshold=(20, 34), s_threshold=(43, 255), v_threshold=(46, 255))
    # set sobelX and sobelY threshold
    gradx = abs_sobel_thresh(img, orient='x', thresh=(30, 100))
    grady = abs_sobel_thresh(img, orient='y', thresh=(50, 150))
    # set magnitude and direction threshold
    mag = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
    dir = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))

    combined = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    combined[(white == 1) | (yellow == 1) | ((gradx == 1) & (grady == 1)) | ((mag == 1) & (dir == 1))] = 1

    # region of interest
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array([[(0, height), (width, height), (width * 0.6, height * 0.5), (width * 0.4, height * 0.5)]],
                        dtype=np.int32)
    roi_image = region_of_interest(img, vertices)

    return roi_image



def image_preprocess4(img):

    """
        image preprocess version 4
        using: yellow threshold, white threshold, lightness threshold,
                sobelX, sobelY, magnitude gradient, direction gradient,  ROI

        Parameters
        ----------
        img: image (np.array())

        Return
        ----------
        the source points
    """

    # set white and yellow threshold
    white = rgb_threshold(img, r_threshold=(200, 255), g_threshold=(200, 255), b_threshold=(200, 255))
    yellow = hsv_threshold(img, h_threshold=(20, 34), s_threshold=(43, 255), v_threshold=(46, 255))
    lightness = yuv_threshold(img, y_threshold=(0, 230))
    # set sobelX and sobelY threshold
    gradx = abs_sobel_thresh(img, orient='x', thresh=(30, 100))
    grady = abs_sobel_thresh(img, orient='y', thresh=(50, 150))
    # set magnitude and direction threshold
    mag = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
    dir = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))

    combined = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    combined[(white == 1) | (yellow == 1) & (lightness == 1) | ((gradx == 1) & (grady == 1)) | ((mag == 1) & (dir == 1))] = 1

    # region of interest
    height = img.shape[0]
    width = img.shape[1]
    vertices = np.array([[(0, height), (width, height), (width * 0.6, height * 0.5), (width * 0.4, height * 0.5)]],
                        dtype=np.int32)
    roi_image = region_of_interest(img, vertices)

    return roi_image


if __name__ == "__main__":

    src = np.float32(([279, 720], [1092, 720], [897, 576], [483, 576]))
    dst = np.float32(([384, 720], [896, 720], [896, 600], [384, 600]))
    original_img = mpimg.imread('test_images/test6.jpg')

    camera_coeff = pickle.load(open("camera_cal/camera_coeff.p", "rb"))
    mtx = camera_coeff["mtx"]
    dist = camera_coeff["dist"]
    undistored_img = cv2.undistort(original_img, mtx, dist, None, mtx)

    image_quality(undistored_img)

    # 2 process the image, get the clear edge information by using color/graident threshold methods(sobel, HLS, and so on...)
    edged_image = image_preprocess1(undistored_img)
    edged_image1 = image_preprocess2(undistored_img)
    edged_image2 = image_preprocess3(undistored_img)


    # generate the projection matrixi
    M, M_inv = transform_perspective(src, dst)

    # 3 translate from the edged image from perspective view to bird's eye view
    edged_warped_img = cv2.warpPerspective(edged_image, M, (original_img.shape[1], original_img.shape[0]))
    edged_warped_img1 = cv2.warpPerspective(edged_image1, M, (original_img.shape[1], original_img.shape[0]))
    edged_warped_img2 = cv2.warpPerspective(edged_image2, M, (original_img.shape[1], original_img.shape[0]))

    plt.imshow(edged_warped_img1, cmap='gray')
    plt.show()

    fig, axes = plt.subplots(3, 3)
    axes[0, 0].imshow(original_img)
    axes[0, 1].imshow(edged_image, cmap='gray')
    axes[0, 2].imshow(edged_warped_img, cmap='gray')
 #   axes[1, 0].imshow(edged_warped_img, cmap='gray')
    axes[1, 1].imshow(edged_image1, cmap='gray')
    axes[1, 2].imshow(edged_warped_img1, cmap='gray')
#    axes[2, 0].imshow(edged_warped_img, cmap='gray')
    axes[2, 1].imshow(edged_image2, cmap='gray')
    axes[2, 2].imshow(edged_warped_img2, cmap='gray')
    plt.show()

    image = mpimg.imread('test_images/test6.jpg')
#    image = region_of_interest(original_image)
    # ksize = 3 # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', thresh=(30, 120))
    grady = abs_sobel_thresh(image, orient='y', thresh=(30, 100))
    mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

    gradient_combined = np.zeros_like(dir_binary)
    gradient_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    white = rgb_threshold(image, g_threshold=(205, 255), b_threshold=(205, 255), r_threshold=(205, 255))
    #yellow = hsv_threshold(image, h_threshold=(50, 110), s_threshold=(50, 255), v_threshold=(50, 255))
    yellow = hsv_threshold(image, h_threshold=(20, 30), s_threshold=(100, 255), v_threshold=(100, 255))
    color_combined = np.zeros_like(white)
    color_combined[(white == 1) | (yellow == 1)] = 1

    f, axes = plt.subplots(3, 3, figsize=(16, 9))
    f.tight_layout()

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=20)

    axes[0, 1].imshow(white, cmap='gray')
    axes[0, 1].set_title('White Image', fontsize=20)

    axes[0, 2].imshow(yellow, cmap='gray')
    axes[0, 2].set_title('Yellow Image', fontsize=20)

    axes[1, 0].imshow(gradx, cmap='gray')
    axes[1, 0].set_title('SobelX Image', fontsize=20)

    axes[1, 1].imshow(grady, cmap='gray')
    axes[1, 1].set_title('SobelY Image', fontsize=20)

    axes[1, 2].imshow(mag_binary, cmap='gray')
    axes[1, 2].set_title('Magnitude Image', fontsize=20)

    axes[2, 0].imshow(dir_binary, cmap='gray')
    axes[2, 0].set_title('Direction Image', fontsize=20)

    axes[2, 1].imshow(color_combined, cmap='gray')
    axes[2, 1].set_title('Color combined Image', fontsize=20)


    axes[2, 2].imshow(gradient_combined, cmap='gray')
    axes[2, 2].set_title('Graident_combined Image', fontsize=20)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    plt.show()
