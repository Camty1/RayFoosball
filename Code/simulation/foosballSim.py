import pybullet as p
import pybullet_data as pd
import time
import math

p.connect(p.GUI)

shapeID = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName="stls/table.stl",
)
bodyID = p.createMultiBody(
    baseCollisionShapeIndex=shapeID,
    basePosition=(0,0,0),
    baseOrientation=(0,0,0,1),
)
