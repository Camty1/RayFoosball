import cv2
import numpy as np

frame2=cv2.imread('/Users/jacklange/ball.png')
frame2=cv2.cvtColor(frame2,cv2.COLOR_BGR2HSV)
hue=frame2[:,:,0] 
sat=frame2[:,:,1] 
val=frame2[:,:,2] 
import numpy as np
huetarget=np.mean(hue)
sattarget=np.mean(sat)
valtarget=np.mean(val)
tolerance=0.15
lower1 = np.array([huetarget*(1-tolerance), sattarget*(1-tolerance), valtarget*(1-tolerance)])
upper1 = np.array([huetarget*(1+tolerance), sattarget*(1+tolerance), valtarget*(1+tolerance)])
xarray=[]
yarray=[]

cap = cv2.VideoCapture("/Users/jacklange/ball.mov")

while True:
    if cap.grab():
        ret, frame = cap.retrieve()
        if not ret:
            continue
        else:
            frame2=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            resultarray = cv2.inRange(frame2, lower1,upper1)
            kernel = np.ones((3,3),np.uint8)
            dilate = cv2.morphologyEx(resultarray, cv2.MORPH_DILATE, kernel, iterations=5)
            indices = np.where(dilate == [255])
            #print("test")
            x=np.mean(indices[0])
            y=np.mean(indices[1])
            xarray.append(x)
            yarray.append(y)
            if np.isnan(x):
                print("can't track")
            else:
                print(x)
                print(y)
                x=int(x)
                y=int(y)
                frame=cv2.circle(frame, (y,x), 20, (0,255,255), 10)
            
            result = cv2.bitwise_and(frame,frame, mask= dilate)
            cv2.imshow('video', frame)
    if cv2.waitKey(10) == 27:
        break
        