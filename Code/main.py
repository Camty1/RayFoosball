#4/22/23
#designed to take everything and integrate into one file
#no clue how to do this but my functions are in here

from computerVision.cv import ball
from computerVision.cv import setCamera

#setCamera()
ball = ball('ball.png')
position = ball.findLocation(0,0)


def runGame():
    position = ball.findLocation(position)

