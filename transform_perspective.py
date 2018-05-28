import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""
It's very important to select src and dst points when you use the transform_perspective function
And for different cameras, this need to choose different src and dst points
"""

SRC_POINTS = np.float32(
    [[581, 477],
     [699, 477],
     [896, 675],
     [384, 675]])

DST_POINTS = np.float32(
    [[384, 200],
     [896, 200],
     [896, 720],
     [384, 720]])

"""
we assume the lane width is 3.7 meter and the dashed lane lines are 3 meters long each in the perspective view
"""
XM_PER_PIXEL = 3.7 / (896-384)
YM_PER_PIXEL = 3.0 / 90


def transform_perspective(src=SRC_POINTS, dst=DST_POINTS):
    """
    We want to tranform from perspective view to bird's eye view,
    so we need to find 4 groups of points in both view images.
    The problem is, we can find 4 points in the image of perspective view,
    but how can we know the corresponding points in the bird's eye view?
    The answer is we can only estimate it. In order to estimate the corresponding points accurately,
    we'd better use the straight lane img. Because if we find 4 points in perspective view properly,
    we can easily make the corresponding points in bird's view look like rectangle.
    For example:
    src_points = np.float32(
    [[581, 477],
     [699, 477],
     [896, 675],
     [384, 675]])

    dst_points = np.float32(
    [[384, 200],
     [896, 200],
     [896, 720],
     [384, 720]])
    """
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv


def warp_pts(pts, M_inv):
    """
        based on equation:

        t * x     [  a11    a12    a13 ]    [  u  ]
        t * y    =|  a21    a22    a23 |    |  v  |
        t         [  a31    a32    a33 }    [  1  ]

        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform
    """
    warped_pts = []
    for pt in pts[0]:
        warped_x = (M_inv[0, 0] * pt[0] + M_inv[0, 1] * pt[1] + M_inv[0, 2]) / \
                       (M_inv[2, 0] * pt[0] + M_inv[2, 1] * pt[1] + M_inv[2, 2])
        warped_y = (M_inv[1, 0] * pt[0] + M_inv[1, 1] * pt[1] + M_inv[1, 2]) / \
                       (M_inv[2, 0] * pt[0] + M_inv[2, 1] * pt[1] + M_inv[2, 2])
        warped_pts.append([warped_x, warped_y])
    return np.array(warped_pts)


def measure_radius_of_curvature(fit_line,  car_y_position=719, ym_per_pixel=YM_PER_PIXEL, xm_per_pixel=XM_PER_PIXEL):
    """
    Measure the radius of curvature in meters.

    We should translate the fit_line from pixel to meters
    if the fit_line in pixel is x = a * y^2 + b * y + c,
    it can be converted to xm_per_pixel * x =  a * (ym_per_pixel * y )^2 + b * (ym_per_pixel * y ) + c in meter.
    As result,  x =  a / xm_per_pixel * (ym_per_pixel * y )^2 + b / xm_per_pixel* (ym_per_pixel * y ) + c / xm_per_pixel
                = Ay^2 + By + C
    then the curvature is (1 + (2Ay+B)^2)^(3/2) / (|2A|), here y is the position of car(image's height)
    https://www.intmath.com/applications-differentiation/8-radius-curvature.php

    :param fit_line:
    :param ym_per_pixel: ? meter = one (x-axis pixel) in bird's eye view image
    :param xm_per_pixle: ? meter = one (y-axis pixel) in bird's eye view image
    :return: left_radius_of_curvature, right_cradius_of_curvature
    """

    left_fit_in_meter = [xm_per_pixel / ym_per_pixel**2 * fit_line[0],
                         xm_per_pixel / ym_per_pixel * fit_line[1],
                         xm_per_pixel * fit_line[2]]

    radius_of_curvature = ((1 + (2 * left_fit_in_meter[0] * car_y_position + left_fit_in_meter[1]) ** 2) ** 1.5) / (2 * left_fit_in_meter[0])

    return radius_of_curvature

def measure_vehicle_offset(left_fit, right_fit, car_position):
    # in pixle

    car_y_position = car_position[0]
    car_x_position = car_position[1]
    lane_bottom_left = left_fit[0] * car_y_position ** 2 + left_fit[1] * car_y_position + left_fit[2]
    lane_bottom_right = right_fit[0] * car_y_position ** 2 + right_fit[1] * car_y_position + right_fit[2]
    vehcile_offset_pixel = (lane_bottom_left + lane_bottom_right) / 2 - car_x_position

    # in meter
    global XM_PER_PIXEL
    vehcile_offset = vehcile_offset_pixel * XM_PER_PIXEL
    return vehcile_offset


if __name__ == "__main__":
    dist_pickle = pickle.load(open("camera_cal/camera_coeff.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    img = mpimg.imread("test_images/straight_lines2.jpg")

    M, M_inv = transform_perspective()
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(warped)
    ax2.set_title('Undistorted and Warped Image', fontsize=30)
    #ax3.imshow(unwarped)
    #ax3.set_title('Unwarped Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

