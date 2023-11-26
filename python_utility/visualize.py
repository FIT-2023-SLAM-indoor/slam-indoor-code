import pybullet as p
import time
import pybullet_data
import math


physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)



file = open("../data/video_report/test.txt",'r')
lst = file.readlines()
file.close()

p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)

p.loadSDF("stadium.sdf")

#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i, val in enumerate(lst):  
    res = lst[i].replace('[', '').replace(']', '').replace(';', '').split(", ")
    if len(res) != 3:
        continue
    if abs(float(res[0])) > 1000 or abs(float(res[1])) > 1000:
        continue
    startPos = [float(res[0]),float(res[1]),float(res[2]) + 3]

    startOrientation = p.getQuaternionFromEuler([0,0,0])
    boxId = p.loadURDF("cube.urdf", startPos, startOrientation, globalScaling=0.12)
    time.sleep(1./240.)


for phi in range(0, 181, 10):
    for theta in range(0, 361, 18):
        x = 2.2 * math.sin(math.radians(phi)) * math.cos(math.radians(theta)) 
        y = 2.2 * math.sin(math.radians(phi)) * math.sin(math.radians(theta))
        z = 2.2 * math.cos(math.radians(phi)) + 3
    
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        boxId = p.loadURDF("sphere2red.urdf", [x, y, z], startOrientation, globalScaling=0.12)
    
    time.sleep(1./240.)

while(1):
    keys = p.getKeyboardEvents()
    cam = p.getDebugVisualizerCamera()
    #Keys to change camera
    if keys.get(109):  #M
        xyz = cam[11]
        x= float(xyz[0]) + 0.125
        y = xyz[1]
        z = xyz[2]
    p.resetDebugVisualizerCamera(cameraYaw = cam[8], cameraPitch= cam[9],cameraDistance = cam[10],cameraTargetPosition=[x,y,z])
    if keys.get(97):   #A
        xyz = cam[11]
        x= float(xyz[0]) - 0.125
        y = xyz[1]
        z = xyz[2]
    p.resetDebugVisualizerCamera(cameraYaw = cam[8], cameraPitch= cam[9],cameraDistance = cam[10],cameraTargetPosition=[x,y,z])
    if keys.get(99):   #C
        xyz = cam[11]
        x = xyz[0] 
        y = float(xyz[1]) + 0.125
        z = xyz[2]
    p.resetDebugVisualizerCamera(cameraYaw = cam[8], cameraPitch= cam[9],cameraDistance = cam[10],cameraTargetPosition=[x,y,z])
    if keys.get(102):  #F
        xyz = cam[11]
        x = xyz[0] 
        y = float(xyz[1]) - 0.125
        z = xyz[2]
        p.resetDebugVisualizerCamera(cameraYaw = cam[8], cameraPitch= cam[9],cameraDistance = cam[10],cameraTargetPosition=[x,y,z])