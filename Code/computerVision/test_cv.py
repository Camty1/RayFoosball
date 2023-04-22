
#cv

from picamera import PiCamera
from time import sleep
import cv2
import numpy as np
import os

path = './computerVision/images/'
for name in os.listdir(path):
    os.remove(os.path.join(path,name))s

camera = PiCamera()
camera.resolution = (1280,720)
camera.framerate = 60

camera.start_preview()
#sleep(5)
camera.stop_preview()

#camera.capture('../../Desktop/test2.png')

for i in range(10):
    fileName = path + "image{}.jpg".format(i)
    camera.capture(fileName)


# Read the input image
img = cv2.imread('table_image.jpg')

# Define the coordinates of the four corners of the table in the input image
src_points = np.array([(50, 50), (450, 50), (50, 250), (450, 250)])

# Define the coordinates of the four corners of the foosball table in the output image
dst_points = np.array([(0, 0), (720, 0), (0, 360), (720, 360)])

# Compute the homography matrix
H, _ = cv2.findHomography(src_points, dst_points)

# Rectify the input image using the homography matrix
rectified_img = cv2.warpPerspective(img, H, (720, 360))

# Display the rectified image
cv2.imshow('Rectified Image', rectified_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

    