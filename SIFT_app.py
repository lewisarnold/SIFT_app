#!/usr/bin/env python

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import numpy as np

import cv2
import sys


class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False
        self._is_set_up = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        self.sift = None
        self.flann = None
        self.template_path = None

        self.template_image = None
        self.template_gray = None

        self.template_key_points = None
        self.template_descriptors = None

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)
        self._is_template_loaded = True

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height,
                             bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        # Code and inspiration taken from:
        # https://pysource.com/2018/06/05/object-tracking-using-homography-opencv-3-4-with-python-3-tutorial-34/
        ret, frame = self._camera_device.read()

        # If no template, just display the webcam feed
        if not self._is_template_loaded:
            pixmap = self.convert_cv_to_pixmap(frame)
            self.live_image_label.setPixmap(pixmap)

        else:
            # if not set up, set things up
            if not self._is_set_up:
                # Load the template image
                self.template_image = cv2.imread(self.template_path)
                self.template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)

                # Make a SIFT object
                self.sift = cv2.xfeatures2d.SIFT_create()
                self.template_key_points, self.template_descriptors = self.sift.detectAndCompute(self.template_gray,
                                                                                                 None)
                # Make a flann object
                index_params = dict(algorithm=0, trees=5)
                search_params = dict()
                self.flann = cv2.FlannBasedMatcher(index_params, search_params)

                self._is_set_up = True

            # Analyze train image with SIFT
            frame_key_points, frame_descriptors = self.sift.detectAndCompute(frame, None)
            # frame = cv2.drawKeypoints(frame, frame_key_points, frame)

            # Flann gets points that are close (in 128 dimensional space)
            matches = self.flann.knnMatch(self.template_descriptors, frame_descriptors, k=2)

            # We select points which are close to each other in distance
            good_points = []
            for m, n in matches:
                # The coefficient below determines the threshold.  Lower number makes it harder to be matched
                # (more restrictive)
                if m.distance < 0.6 * n.distance:
                    good_points.append(m)

            image_with_matches = cv2.drawMatches(self.template_gray, self.template_key_points, frame, frame_key_points,
                                                 good_points, frame)

            # If we have more than a certain number of good points, we can make a homography
            if len(good_points) > 10:
                # extract the actual point locations from the list of points
                query_pts = np.float32([self.template_key_points[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                train_pts = np.float32([frame_key_points[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

                # Find a consensus on which points map to which using RANSAC
                # Returns the perspective transform which needs to be applied to get from one image to the other
                perspective_transform_matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

                # Do the perspective transform
                height = self.template_image.shape[0]
                width = self.template_image.shape[1]

                # The corners of the template image
                pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
                # Apply the perspective transform to get the corners in the train image
                dst = cv2.perspectiveTransform(pts, perspective_transform_matrix)

                # Draw the frame around the template as it appears in the train image
                homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
                out = homography

            else:
                out = image_with_matches

            pixmap = self.convert_cv_to_pixmap(out)
            self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
