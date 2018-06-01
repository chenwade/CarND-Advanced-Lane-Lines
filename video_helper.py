import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip, CompositeVideoClip
import os

frame_num = 0


def obtain_sub_video(input_video, output_video, start_seconds, end_seconds):
    """
    get a part of the input video, the output playing content of input video from start_seconds to end_seconds

    Parameters
    ----------
    input video: the long video need to be clipped
    output video: the short video which is the subset of long video
    start_seconds: the begin time(s) of long video
    end_seconds:  the end time(s) of long video

    Return
    ----------
    the output video

    """
    video = VideoFileClip(input_video).subclip(start_seconds, end_seconds)
    result = CompositeVideoClip([video, ])
    result.to_videofile(output_video)


def divide_video_into_images(input_video, image_folder, format='jpg'):
    """
    divide the video to images and store the images into the givin folders

     Parameters
    ----------
    input video: the video need to be processed
    image_folder: the path to store the images(jpg format)

    Return
    ----------
    None
    """

    assert format == 'jpg' or format == 'jpeg' or format == 'JPG' or format == 'png' or format == 'PNG'

    # the this function, each frame of the video will be stored into the folder
    def store_image(img):
        """
        store the each frmae image into the folder
        """
        global frame_num
        save_path = image_folder
        frame_name = save_path + ("%d." % frame_num) + format
        mpimg.imsave(frame_name, img, format=format)
        frame_num += 1
        return img

    # it will generate a output video based on the output of 'store_image' function
    # but we don't need it, we only care the process that store the image.
    clip = VideoFileClip(input_video)
    video_clip = clip.fl_image(store_image)
    _ = 'out.mp4'
    video_clip.write_videofile(_, audio=False)
    os.remove(_)


if __name__ == "__main__":
    obtain_sub_video('harder_challenge_video.mp4', 'short_harder_challenge_video.mp4', 0, 10)
    divide_video_into_images('harder_challenge_video.mp4', 'debug_folder/')