import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip, CompositeVideoClip

frame = 0


def obtain_sub_video(input_video, output_video, start_seconds, end_seconds):
    """
    get the part of the input video, the output is the input video from start_seconds to end_seconds
    """
    video = VideoFileClip(input_video).subclip(start_seconds, end_seconds)
    result = CompositeVideoClip([video, ])
    result.to_videofile(output_video)


def store_image(img):
    """
    store the image to a folder
    """
    global frame_num
    save_path = "/home/wade/data/advance_lane_debug_data/"
    frame_name = save_path + ("%d.jpg" % frame_num)
    mpimg.imsave(frame_name, img, format='jpg')
    frame_num += 1
    return img


def divide_video_2_image(input_video, output_video):
    """
    divide the video to images
    """
    clip = VideoFileClip(input_video)
    video_clip = clip.fl_image(store_image)
    video_clip.write_videofile(output_video, audio=False)


if __name__ == "__main__":
    obtain_sub_video('harder_challenge_video.mp4', 'short_harder_challenge_video.mp4', 0, 10)