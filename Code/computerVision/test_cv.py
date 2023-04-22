
#cv

from picamera import PiCamera
from time import sleep
import cv2
import numpy as np
import os

path = './computerVision/images/'
for name in os.listdir(path):
    os.remove(os.path.join(path,name))

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
    