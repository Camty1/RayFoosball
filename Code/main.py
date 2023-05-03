#4/22/23
#designed to take everything and integrate into one file
#no clue how to do this but my functions are in here

from computerVision.cvCopy import ball
from computerVision.cvCopy import setCamera
import time

#setCamera()
ball = ball('./Code/ball.png')
tic = time.perf_counter()
for i in range(10):
    position = ball.findLocation(0,0)
toc = time.perf_counter()
print({toc-tic})

def runGame():
    position = ball.findLocation(position)

