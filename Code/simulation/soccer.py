import pybullet as p
import pybullet_data as pd
import time
import math

theta = math.pi/32

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
offset = 0
for scale in range (1,2,1):
  ball = p.loadURDF("soccerball.urdf",[0,0,offset + 1], globalScaling=scale*0.1)
  p.changeDynamics(ball,-1,linearDamping=0, angularDamping=0, rollingFriction=0.001, spinningFriction=0.001, restitution=1/math.sqrt(2))
  p.resetBaseVelocity(ball, (5,0,0), (0,0,0,1))
  p.changeVisualShape(ball,-1,rgbaColor=[0.8,0.8,0.8,1])
  offset += 2*scale*0.1
planeId = p.loadURDF("plane.urdf", (0,0,0), (0, -math.sin(theta/2), 0, math.cos(theta/2)))
print(p.getDynamicsInfo(planeId, -1))
p.changeDynamics(planeId, -1, restitution=1)
p.setGravity(0,0,-10)
f = open("./simulation/soccer_output.csv", "w")
t0 = time.time()
for i in range (240*10+1):
  #p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL)
  # f.write(str(i) + ":", str(p.getLinkState(ball, -1)))
  # f.write("\n")
  r = p.getBasePositionAndOrientation(ball)
  x,y,z = r[0]
  a,b,c,d = r[1]

  f.write(f"{1/240. * i}, {x}, {y}, {z}, {a}, {b}, {c}, {d}")
  f.write("\n")
  p.stepSimulation()
  time.sleep(1/240.)

f.close()
p.disconnect()