import cv2
import numpy as np


# Author: Mostafa Mahmoud
# Email: mostafa_mahmoud@protonmail.com
# Created on: 12/16/15

class HandTracker:
    DEBUGGING = True
    SPACE_KEY = 32
    ESC_KEY = 27
    camera = None
    frame = None
    hand_row_point = None
    hand_col_point = None
    hand_row_opposite_point = None
    hand_col_opposite_point = None
    hand_histogram_captured = None
    hand_histogram = None
    skin_mask = None

    def __init__(self):
        self.hand_histogram_captured = False

    def start(self):
        self.start_camera()
        while self.camera.isOpened():
            if not self.hand_histogram_captured:
                self.capture_hand_histogram()
            else:
                # get feed from camera frame by frame
                _, self.frame = self.camera.read()
                self.apply_skin_mask(self.frame)
                cv2.imshow('skin mask', self.skin_mask)
                # Wait for a key press
                pressed_key = cv2.waitKey(16)
                if pressed_key == self.ESC_KEY:
                    cv2.destroyAllWindows()
                    break

    def start_camera(self):
        self.camera = cv2.VideoCapture(0)

    def draw_hand_filter(self, frame):
        rows, cols, _ = frame.shape
        # Place 9 points at the center of the image
        self.hand_row_point = np.array(
                [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 10 * rows / 20, 10 * rows / 20, 10 * rows / 20,
                 14 * rows / 20, 14 * rows / 20, 14 * rows / 20])
        self.hand_col_point = np.array(
                [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20,
                 9 * cols / 20, 10 * cols / 20, 11 * cols / 20])
        # another 9 points at the other corner of the squares we want to draw
        # 10 is the width of the square
        self.hand_row_opposite_point = self.hand_row_point + 10
        self.hand_col_opposite_point = self.hand_col_point + 10
        # draw the squares blue colored (255,0,0)
        size = self.hand_row_point.size
        for i in xrange(size):
            cv2.rectangle(frame, (self.hand_col_point[i], self.hand_row_point[i]),
                          (self.hand_col_opposite_point[i], self.hand_row_opposite_point[i]),
                          (255, 0, 0), 1)
        return frame

    def capture_hand_histogram(self):
        assert self.camera.isOpened()
        if self.hand_histogram_captured:
            return
        # get feed from camera frame by frame
        _, self.frame = self.camera.read()
        # Draw hand detection filter
        result = self.draw_hand_filter(self.frame)
        # Show the edited frame
        cv2.imshow('Cover all squares with your hand', result)
        # Wait for a key press
        pressed_key = cv2.waitKey(16)
        if pressed_key == self.SPACE_KEY:
            self.scan_skin()
            cv2.destroyWindow('Cover all squares with your hand')
            self.hand_histogram_captured = True

    def scan_skin(self):
        # transform from rgb to hsv because it's easier
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        # roi (region of interest) 900 Points (9 squares each of size 10 * 10)
        roi = np.zeros([90, 10, 3], dtype=hsv.dtype)
        size = self.hand_row_point.size
        for i in xrange(size):
            roi[i * 10:i * 10 + 10, 0:10] = hsv[self.hand_row_point[i]:self.hand_row_point[i] + 10,
                                            self.hand_col_point[i]:self.hand_col_point[i] + 10]

        self.hand_histogram = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(self.hand_histogram, self.hand_histogram, 0, 255, cv2.NORM_MINMAX)

    def apply_skin_mask(self, frame):
        # transform from rgb to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # detect skin using the hand histogram
        self.skin_mask = cv2.calcBackProject([hsv], [0, 1], self.hand_histogram, [0, 180, 0, 256], 1)
        # create a elliptical kernel (11 is the best in my case)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        cv2.filter2D(self.skin_mask, -1, kernel, self.skin_mask)
        # Apply gaussian filter to give much better result
        cv2.GaussianBlur(self.skin_mask, (3, 3), 0, self.skin_mask)
        # change the threshold to suit the brightness (20-30 gave me best results so far)
        _, thresh = cv2.threshold(self.skin_mask, 20, 255, 0)
        thresh = cv2.merge((thresh, thresh, thresh))
        # Mask the hand from the original frame
        self.skin_mask = cv2.bitwise_and(frame, thresh)
        # Apply gaussian filter to give much cleaner result
        cv2.GaussianBlur(self.skin_mask, (5, 5), 0, self.skin_mask)
        # remove faulty skin (kernel of size 9x9)
        cv2.morphologyEx(self.skin_mask, cv2.MORPH_OPEN, (31, 31), self.skin_mask, iterations=5)
        # reduce black spaces in the hand
        cv2.morphologyEx(self.skin_mask, cv2.MORPH_CLOSE, (9, 9), self.skin_mask, iterations=5)
        return self.skin_mask

if __name__ == '__main__':
    HandTracker().start()
