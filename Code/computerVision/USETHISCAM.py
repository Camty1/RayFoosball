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
        tolerance=0.4
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
                    
    def read(self,prevx,prevy):
        # return the frame most recently read
        # cv2.imshow("Ron Johnson", self.frame)

        #convert frame to HSV
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        #define range of color in HSV
        lower = self.lower1
        upper = self.upper1
        #threshold the HSV image to get only the color
        mask = cv2.inRange(hsv, lower, upper)
        #bitwise-AND mask and original image
        res = cv2.bitwise_and(self.frame,self.frame, mask= mask)
        #find contours in the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
        #initialize center of ball
        center = None
        #only proceed if at least one contour was found
        if len(cnts) > 0:
            #find the largest contour in the mask
            c = max(cnts, key=cv2.contourArea)
            #find the minimum enclosing circle
            ((x,y), radius) = cv2.minEnclosingCircle(c)
            #find the moments of the contour
            M = cv2.moments(c)
            #calculate the center of the contour
            if M["m00"] != 0:
                x = int(M["m10"]/M["m00"])
                y = int(M["m01"]/M["m00"])
                x_center = x-180
                y_center = y-120
                velocityx = x_center-prevx
                velocityy = y_center-prevy
                return x_center,y_center,velocityx,velocityy
            else:
                print("cannot track")
                return prevx,prevy
        
    
    def stop(self):
         # indicate that the thread should be stopped
         self.stopped = True
        


vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()
frames = []
# loop over some frames...this time using the threaded stream
while fps._numFrames < 100:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    x,y,vx,vy= vs.read()
    print(x,y,vx,vy)
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

#width, height = 360, 240
#fps = 30
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#print(frames)
# create the video writer object
#out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
#for frame in frames:
    #print("writing frame")
    #out.write(frame)
#print(frames[0])
#out.release()
#print("done")