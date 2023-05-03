# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import cv2
import numpy as np

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import time

class PiVideoStream:
    def __init__(self, resolution=(360,240), framerate=32,color='green.png'):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                format="bgr", use_video_port=True)
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        self.color = cv2.imread(color)
        self.color = cv2.cvtColor(self.color,cv2.COLOR_BGR2HSV)
        hue=self.color[:,:,0] 
        sat=self.color[:,:,1] 
        val=self.color[:,:,2]
        huetarget=np.mean(hue)
        sattarget=np.mean(sat)
        valtarget=np.mean(val)
        tolerance=0.2
        self.lower1 = np.array([huetarget*(1-tolerance), sattarget*(1-tolerance), valtarget*(1-tolerance)])
        self.upper1 = np.array([huetarget*(1+tolerance), sattarget*(1+tolerance), valtarget*(1+tolerance)])
        self.xarray=[]
        self.yarray=[]
            
    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return
                    
    def read(self):
        # return the frame most recently read
        # cv2.imshow("Ron Johnson", self.frame)
        cv2.cvtColor(self.frame,cv2.COLOR_RGB2BGR)
        cv2.cvtColor(self.frame,cv2.COLOR_BGR2HSV)
        resultarray = cv2.inRange(self.frame, self.lower1,self.upper1)
        kernel = np.ones((3,3),np.uint8)
        dilate = cv2.morphologyEx(resultarray, cv2.MORPH_DILATE, kernel, iterations=5)
        indices = np.where(dilate == [255]) 
        if not indices[0].any() or not indices[1].any():
            print( "can't track")
        else:
            x=np.mean(indices[0])
            y=np.mean(indices[1])
            self.xarray.append(x)
            self.yarray.append(y)
            print(x, y)
            x=int(x)
            y=int(y)
            cv2.circle(self.frame, (y,x), 20, (0,255,255), 10)
        return self.frame
    
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        


vs = PiVideoStream().start()
time.sleep(2.0)
frame = vs.read()
cv2.imwrite("InitialImage.jpg", frame)
fps = FPS().start()
# loop over some frames...this time using the threaded stream
while fps._numFrames < 1000:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # check to see if the frame should be displayed to our screen
    # cv2.imshow("Ron Johnson", frame)
    key = cv2.waitKey(1) & 0xFF 
    # update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
