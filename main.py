import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

import argparse
import re
import time
import sys

from lane import *
from calibrate_camera import *


def detect_ego_lane(original_img):
    """
    detect lane and annotate the lane area in the original image

    Parameters
    ----------
    original_img: the input image or a image frame of video

    Return
    ----------
    the annotated image

    """

    global road_lane
    global camera_coeff
    global file_type

    # detect lane and annotate the lane area in the original image
    if file_type == 'video':
        annotated_img = road_lane.video_detect(original_img, camera_coeff)
    if file_type == 'image':
        annotated_img = road_lane.image_detect(original_img, camera_coeff)
    return annotated_img


if __name__ == "__main__":
    # set defalut parameter
    parser = argparse.ArgumentParser(prog='main.py', usage='python %(prog)s -i input_file -o [output_file]',
                                     description='detect lane from images or pictures')
    parser.add_argument('-i', '--input_file', type=str, default='output_images/undistored.jpg',
                        help='input image or video file to process')
    parser.add_argument('-o', '--output_file', type=str, default='harder_challenge_video_out.mp4', help='processed image or video file')
    args = parser.parse_args()

    """
    .可以匹配任意字符
    .+ 至少一个任意字符
    ^.+ 开始至少一个任意字符
    ^.+\.mp4$  开始至少一个任意字符且.mp4结尾
    """

    # check whether the input file is video or image
    video_pattern = re.compile("^.+\.mp4$")
    image_pattern = re.compile("^.+\.(jpg|jpeg|JPG|png|PNG)$")

    if video_pattern.match(args.input_file):
        if not os.path.exists(args.input_file):
            print("Video input file: %s does not exist.  Please check and try again." % (args.input_file))
            sys.exit(1)
        else:
            file_type = 'video'
    elif image_pattern.match(args.input_file):
        if not os.path.exists(args.input_file):
            print("Image input file: %s does not exist.  Please check and try again." % (args.input_file))
            sys.exit(2)
        else:
            file_type = 'image'
    else:
        print("Invalid video/image filename extension for output.  Must end with '.mp4', '.jpg' '.png'... ")
        sys.exit(3)

    # calibrate the camera
    calibrate_camera()
    # get the coefficients of camera
    camera_coeff = pickle.load(open("camera_cal/camera_coeff.p", "rb"))

    # debug == True: normal mode
    # debug == False: debug mode, this mode will show lots of debug info
    road_lane = Lanes(debug=True)

    # process image
    if file_type == 'image':
        print("image processing %s..." % (args.input_file))
        start_time = time.clock()

        img = plt.imread(args.input_file)
        result = detect_ego_lane(img)

        end_time = time.clock()
        print("running time %s seconds" % (end_time - start_time))

        plt.imshow(result)
        plt.xlim(0, result.shape[1])
        plt.ylim(result.shape[0], 0)
        plt.show()

    # process video
    if file_type == 'video':
        print("video processing %s..." % (args.input_file))
        start_time = time.clock()
        frame_num = 0
        clip1 = VideoFileClip(args.input_file)
        video_clip = clip1.fl_image(detect_ego_lane)
        video_clip.write_videofile(args.output_file, audio=False)

        end_time = time.clock()
        print("running time %s seconds" % (end_time - start_time))





