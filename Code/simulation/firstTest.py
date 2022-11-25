import pybullet as p
import time
import pybullet_data
import numpy as np
#import torch as pt
import math

pi = math.pi


dt = 0.01

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = [0,0,0,1]
boxId = p.loadURDF("r2d2.urdf",startPos, startOrientation)
print(p.getNumJoints(boxId))
for i in range(15):
    print(p.getJointInfo(boxId, i))
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range (1000):
    # p.setJointMotorControl2(boxId, 2, controlMode=p.VELOCITY_CONTROL, targetVelocity = 25, force= 10)
    # p.setJointMotorControl2(boxId, 3, controlMode=p.VELOCITY_CONTROL, targetVelocity = 25, force= 10)
    # p.setJointMotorControl2(boxId, 6, controlMode=p.VELOCITY_CONTROL, targetVelocity = -25, force= 10)
    # p.setJointMotorControl2(boxId, 7, controlMode=p.VELOCITY_CONTROL, targetVelocity = -25, force= 10)
    p.setJointMotorControl2(boxId, 1, controlMode=p.VELOCITY_CONTROL, targetVelocity = 10, force = 10)
    p.stepSimulation()
    time.sleep(dt)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
