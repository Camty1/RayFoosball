
#cv

from picamera import PiCamera
from time import sleep
#import opencv as cv

camera = PiCamera()

camera.start_preview()
sleep(5)
camera.stop_preview()

camera.capture('Desktop/test.jpg')