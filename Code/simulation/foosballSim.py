import pybullet as p
import pybullet_data as pd
import time
import numpy as np
import math
import random

runTime = 10 # seconds
goals = [0,0]

PI = math.pi

ball_offset = .32

## Reset simulation to initial state after someone scores
# Right now just puts ball in middle of table and sends towards a goal
# but will reset players and start ball from side of table
def resetSim(goal=0):

    angle = random.gauss(0, 10) * PI / 180
    
    position = [0, .3, .1]
    velocity = [.5*math.sin(angle), .5*math.cos(angle), 0]

    
    if goal == goal2:
        position[1] = -position[1]
        velocity = [ -x for x in velocity ]
    
    p.resetBasePositionAndOrientation(ball, position, (0,0,0,1))
    p.resetBaseVelocity(ball, velocity, (0,0,0,1))
    resetPlayers()
    p.performCollisionDetection()
    contact = p.getContactPoints(ball)
    print(contact)

def resetPlayers():
    for joint in (prismaticList + rotationList):
        p.resetJointState(table, joint, 0, 0)
    resetControllers()

def resetControllers():
    zeroList = np.zeros(len(prismaticList)).tolist()
    p.setJointMotorControlArray(bodyUniqueId=table, jointIndices=prismaticList, controlMode=p.POSITION_CONTROL, targetPositions=zeroList)
    p.setJointMotorControlArray(bodyUniqueId=table, jointIndices=rotationList, controlMode=p.POSITION_CONTROL, targetPositions=zeroList)

def convertSetpoint(values, mins, maxes):
    
    setpoints = [];
    
    for i in range(4):
        setpoints.append(map(values[2*i], maxes[i], mins[i]))
        setpoints.append(map(values[2*i+1], maxes[i], mins[i]))

    return setpoints

def map(value, maxv, minv):
    return value * (maxv-minv) + minv

def sinSetpoints(frequency, time):
    value = (1+math.sin(2*PI*frequency*time))*.5
    return np.ones(8)*value

# Returns ((ballP, ballV), (prisP1, rotP1, prisV1, rotV1), (prisP2, rotP2, prisV2, rotV2))
def getState():
    # Ball State
    ballPos = p.getBasePositionAndOrientation(ball)
    ballVel = p.getBaseVelocity(ball)

    # Robot State
    prisState = p.getJointStates(table, prismaticList)
    rotState = p.getJointStates(table, rotationList)

    # State unpacking
    prisPos = [prisState[x][0] for x in range(8)]
    rotPos = [rotState[x][0] for x in range(8)]
    prisVel = [prisState[x][1] for x in range(8)]
    rotVel = [prisState[x][1] for x in range(8)]

    return (ballPos[0][:2] + ballVel[0][:2], tuple(prisPos[::2] + rotPos[::2] + prisVel[::2] + rotVel[::2]), tuple(prisPos[1::2] + rotPos[1::2] + prisVel[1::2] + rotVel[1::2]))

def getReward(state, gains=(10, -10, 10, -10), goalx=1219.2/2000):
    goal1 = np.array([-goalx, 0])
    goal2 = np.array([goalx, 0])

    ballPos = np.array(state[0][:2])
    ballVel = np.array(state[0][2:])

    goal1Vec = goal1 - ballPos
    goal2Vec = goal2 - ballPos

    goal1Dist = np.linalg.norm(goal1Vec)
    goal2Dist = np.linalg.norm(goal2Vec)

    goal1Vel = np.dot(ballVel, goal1Vec)/goal1Dist
    goal2Vel = np.dot(ballVel, goal2Vec)/goal2Dist
    
    return gains[0] * goal1Dist + gains[1] * goal2Dist + gains[2] * goal1Vel + gains[3] * goal2Vel


# Pybullet setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

# Table base position and rotation
tableRotation = (0,0,0,1)
tablePosition = (0,0,0) 

# Goal positions 
goal1Position = (-25.4/1000-1219.2/2000, 0, 25.4/1000)
goal2Position = ((1219.2)/2000, 0, 25.4/1000)

# Loading URDF files
table = p.loadURDF("TableTest.urdf")
ball = p.loadURDF("ball.urdf", basePosition=(0, .3, 0.1))
goal1 = p.loadURDF("Goal.urdf", basePosition=goal1Position)
goal2 = p.loadURDF("Goal.urdf", basePosition=goal2Position)

# Dynamics setup
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(1.5, 50, -35, (0,0,0))

# Joint controls setup
prismaticList = [0, 5, 10, 14, 18, 25, 32, 37]
rotationList = [1, 6, 11, 15, 19, 26, 33, 38]

team1Prismatic = prismaticList[::2]
team1Rotatoin = rotationList[::2]

team2Prismatic = prismaticList[1::2]
team2Rotation = rotationList[1::2]

prismaticMaxes = [.18161, .35306, .11176, .22606]
rotationMaxes = [PI, PI, PI, PI]
prismaticMins = [0, 0, 0, 0]
rotationMins = [-PI, -PI, -PI, -PI]


counter = 0

resetSim()

# Run simulation
for i in range(240 * runTime):
       # p.setJointMotorControlArray(bodyUniqueId=table, jointIndices=prismaticList, controlMode=p.POSITION_CONTROL, targetPositions=convertSetpoint(sinSetpoints(.25, i/240), prismaticMins, prismaticMaxes))
        #p.setJointMotorControlArray(bodyUniqueId=table, jointIndices=rotationList, controlMode=p.POSITION_CONTROL, targetPositions=convertSetpoint(sinSetpoints(.5, i/240), rotationMins, rotationMaxes))

    if i%240 == 0:
        state = getState()
        reward1 = getReward(state)
        reward2 = -reward1
        print(reward1, reward2)

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

