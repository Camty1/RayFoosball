import pybullet as p
import pybullet_data as pd
import time
import math
import os

runTime = 10 # seconds
goals = [0,0]

## Reset simulation to initial state after someone scores
# Right now just puts ball in middle of table and sends towards a goal
# but will reset players and start ball from side of table
def resetSim(goal):
    p.resetBasePositionAndOrientation(ball, (0,0,.2), (0,0,0,1)) 
    if goal == goal1:
        p.resetBaseVelocity(ball, (.5,0,0), (0,0,0,1))
    else:
        p.resetBaseVelocity(ball, (-.5,0,0), (0,0,0,1))


# Pybullet setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

# Table base position and rotation
tableRotation = (0,0,0,1) #(math.sin(math.pi/4),0,0,math.cos(math.pi/4))
tablePosition = (0,0,0) #(-(1219.2)/2/1000, (730.250000)/2/1000, 0)

# Goal positions 
goal1Position = (-25.4/1000-1219.2/2000, 0, 25.4/1000)
goal2Position = ((1219.2)/2000, 0, 25.4/1000)

# Loading URDF files
table = p.loadURDF("CollisionTable.urdf", baseOrientation=tableRotation, basePosition=tablePosition)
ball = p.loadURDF("ball.urdf", basePosition=(0, 0, 0.2))
goal1 = p.loadURDF("Goal.urdf", baseOrientation=tableRotation, basePosition=goal1Position)
goal2 = p.loadURDF("Goal.urdf", baseOrientation=tableRotation, basePosition=goal2Position)

# Dynamics setup
p.resetBaseVelocity(ball, (.5, 0, 0), (0,0,0,1))
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(1.5, 50, -35, (0,0,0))

# Run simulation
for i in range(240 * runTime):
    ballState = p.getBasePositionAndOrientation(ball) # Get ball position
    p.stepSimulation()
    
    # Check for goals (1 scores if it hits goal 2 and vice versa)
    score1 = p.getContactPoints(goal2)
    score2 = p.getContactPoints(goal1)

    # If either list is not empty, then we know someone has scored
    if score1:
        goals[0] += 1
        resetSim(goal2)
    if score2:
        goals[1] += 1
        resetSim(goal1)

    time.sleep(1/240.)

print(goals)
# End simulation
p.disconnect()
