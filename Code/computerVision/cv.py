#Jack Lange
#4/22/23
#initial build of the python ball class

from picamera import PiCamera
import cv2
import colorsys
import numpy as np

def setCamera(self, camera):
        camera.resolution = (1280,720)
        camera.framerate = 60

class ball():
    def __init__(self, color):
        self.camera = PiCamera()
        self.camera.start_preview()
        self.color=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
        hue=self.color[:,:,0] 
        sat=self.color[:,:,1] 
        val=self.color[:,:,2]
        huetarget=np.mean(hue)
        sattarget=np.mean(sat)
        valtarget=np.mean(val)
        tolerance=0.15
        self.lower1 = np.array([huetarget*(1-tolerance), sattarget*(1-tolerance), valtarget*(1-tolerance)])
        self.upper1 = np.array([huetarget*(1+tolerance), sattarget*(1+tolerance), valtarget*(1+tolerance)])
        self.xarray=[]
        self.yarray=[]

    def findLocation(self,oldx,oldy):
        frame = self.camera.capture()
        resultarray = cv2.inRange(frame, self.lower1,self.upper1)
        kernel = np.ones((3,3),np.uint8)
        dilate = cv2.morphologyEx(resultarray, cv2.MORPH_DILATE, kernel, iterations=5)
        indices = np.where(dilate == [255])            
        #print("test")
        x=np.mean(indices[0])
        y=np.mean(indices[1])
        self.xarray.append(x)
        self.yarray.append(y)
        if np.isnan(x):
            return "can't track"
        else:
            print(x)
            print(y)
            x=int(x)
            y=int(y)
            frame=cv2.circle(frame, (y,x), 20, (0,255,255), 10)
        
            point1 = np.array([oldx, oldy])
            point2 = np.array([x, y])

            # Calculate velocity vector
            velocity = point2 - point1
            return (x,y,velocity)

    