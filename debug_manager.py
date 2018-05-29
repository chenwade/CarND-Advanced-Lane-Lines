import cv2
import numpy as np
import matplotlib.pyplot as plt


class DebugManager():
    def __init__(self):

        self.frame_num = 0
        #input image
        self.original_image = None
        #edged image
        self.edged_image = None
        #warped image(bird_view)
        self.warped_image = None
        #edged and warped image
        self.edged_warped_image = None
        #lane fit image in bird view
        self.lane_fit_image = None
        #debug lane annotate
        self.lane_annotate_image = None

        # x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        #left fit line and right fit line
        self.left_fit = None
        self.right_fit = None
        self.adjusted_left_fit = None
        self.adjusted_right_fit = None
        self.left_fit_valid = False
        self.right_fit_valid = False

        self.parallel_mse = None
        self.left_fit_mse = None
        self.right_fit_mse = None

        # annotate the lane in the area?
        self.annotate = False

        # if current fit passed the sanity check, we can think we detected the correct lanes
        self.sanity_check_result = False
        # 5 recent sanity check results
        self.recent10_check = None
        self.con_detect_num = 0
        self.con_not_detect_num = 0

        #radius_of_curvature
        self.left_radius_of_curvature = None
        self.right_radius_of_curvature = None

        #vehicle offset in ego lane
        self.vehicle_offset = None


    def debug_video_show(self):
        """
        show edged, edged_warped, lane_fit, lane_annotate in one image
        :return: combined image
        """
        # assemble the screen
        debug_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        debug_screen[0:720, 0:1280] = cv2.resize(self.lane_fit_image, (1280, 720), interpolation=cv2.INTER_AREA)
        debug_screen[720:1080, 0:640] = cv2.resize(self.edged_image, (640, 360), interpolation=cv2.INTER_AREA)
        debug_screen[720:1080, 640:1280] = cv2.resize(self.edged_warped_image, (640, 360), interpolation=cv2.INTER_AREA)
        if self.annotate:
            debug_screen[720:1080, 1280:1920] = cv2.resize(self.lane_annotate_image, (640, 360),
                                                           interpolation=cv2.INTER_AREA)
        else:
            debug_screen[720:1080, 1280:1920] = cv2.resize(self.original_image, (640, 360),
                                                           interpolation=cv2.INTER_AREA)

        font = cv2.FONT_HERSHEY_COMPLEX
        color = (128, 128, 0)
        debug_info = np.zeros((640, 720, 3), dtype=np.uint8)
        cv2.putText(debug_info, 'frame num: %d   sanity check: %s' % (self.frame_num, self.sanity_check_result),
                    (30, 60), font, 1, color, 2)
        cv2.putText(debug_info, 'is left fit valid:  %s, mse = %5.1f' % (self.left_fit_valid, self.left_fit_mse), (30, 120), font, 1, color, 2)
        cv2.putText(debug_info, 'left fit: %5.5fy^2 + %5.2fy + %5.1f' %
                    (self.left_fit[0], self.left_fit[1], self.left_fit[2]), (30, 150), font, 1, color, 2)
        cv2.putText(debug_info, 'is right fit valid: %s, mse = %5.1f' % (self.right_fit_valid, self.right_fit_mse), (30, 180), font, 1, color, 2)
        cv2.putText(debug_info, 'right fit: %5.5fy^2 + %5.2fy + %5.1f ' %
                    (self.right_fit[0], self.right_fit[1], self.right_fit[2]), (30, 210), font, 1, color, 2)

        cv2.putText(debug_info, 'mse between left and right fit: %5.1f' % self.parallel_mse, (30, 240), font, 1, color, 2)

        cv2.putText(debug_info, 'last 5 fit: ', (30, 300), font, 1, color, 2)
        cv2.putText(debug_info, '   %s ' % self.recent10_check, (30, 330), font, 1, color, 2)
        cv2.putText(debug_info, 'con detect: %d ' % self.con_detect_num, (30, 360), font, 1, color, 2)
        cv2.putText(debug_info, 'con not detect: %d ' % self.con_not_detect_num, (30, 390), font, 1, color, 2)

        if self.annotate:
            cv2.putText(debug_info, 'left RoC is %5.2f km ' % (self.left_radius_of_curvature / 1000),
                    (30, 450), font, 1, color, 2)
            cv2.putText(debug_info, 'right RoC is %5.2f km ' % (self.right_radius_of_curvature / 1000),
                    (30, 480), font, 1, color, 2)
            cv2.putText(debug_info, 'Vehicle offset from center is %5.2f m' % self.vehicle_offset,
                    (30, 510), font, 1, color, 2)

        debug_screen[0:720, 1280:1920] = cv2.resize(debug_info, (640, 720), interpolation=cv2.INTER_AREA)
        return debug_screen


    def debug_image_show(self):
        """
        show edged, edged_warped, lane_fit, lane_annotate in one image
        :return: combined image
        """
        # assemble the screen
        debug_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        debug_screen[0:720, 0:1280] = cv2.resize(self.lane_fit_image, (1280, 720), interpolation=cv2.INTER_AREA)
        debug_screen[720:1080, 0:640] = cv2.resize(self.edged_image, (640, 360), interpolation=cv2.INTER_AREA)
        debug_screen[720:1080, 640:1280] = cv2.resize(self.edged_warped_image, (640, 360), interpolation=cv2.INTER_AREA)

        debug_screen[720:1080, 1280:1920] = cv2.resize(self.lane_annotate_image, (640, 360),
                                                           interpolation=cv2.INTER_AREA)

        font = cv2.FONT_HERSHEY_COMPLEX
        color = (128, 128, 0)
        debug_info = np.zeros((640, 720, 3), dtype=np.uint8)
        cv2.putText(debug_info, 'sanity check: %s' % self.sanity_check_result, (30, 60), font, 1, color, 2)

        cv2.putText(debug_info, 'left fit: %5.5fy^2 + %5.2fy + %5.1f' %
                    (self.left_fit[0], self.left_fit[1], self.left_fit[2]), (30, 150), font, 1, color, 2)

        cv2.putText(debug_info, 'right fit: %5.5fy^2 + %5.2fy + %5.1f ' %
                    (self.right_fit[0], self.right_fit[1], self.right_fit[2]), (30, 180), font, 1, color, 2)

        cv2.putText(debug_info, 'mse between left and right fit: %5.1f' % self.parallel_mse, (30, 240), font, 1, color, 2)

        cv2.putText(debug_info, 'left RoC is %5.2f km ' % (self.left_radius_of_curvature / 1000),
                    (30, 450), font, 1, color, 2)
        cv2.putText(debug_info, 'right RoC is %5.2f km ' % (self.right_radius_of_curvature / 1000),
                    (30, 480), font, 1, color, 2)
        cv2.putText(debug_info, 'Vehicle offset from center is %5.2f m' % self.vehicle_offset,
                    (30, 510), font, 1, color, 2)

        debug_screen[0:720, 1280:1920] = cv2.resize(debug_info, (640, 720), interpolation=cv2.INTER_AREA)
        return debug_screen

if __name__ == "__main__":
    a = [False, False, False, False, False]
    font = cv2.FONT_HERSHEY_COMPLEX
    color = (128, 128, 0)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    img = cv2.resize(image, (360, 640), interpolation=cv2.INTER_AREA)
    cv2.putText(img, "sanity check: %s" % a, (30, 60),  font, 1, color, 2)
    plt.imshow(img)
    plt.show()