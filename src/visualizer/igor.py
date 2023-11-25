import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)

file = open("../../data/video_report/3Dpoints.txt",'r')
lst = file.readlines()
file.close()


#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range(20000):
    
    res = lst[i].replace('[', '').replace(']', '').replace(';', '').split(", ")
    if len(res) != 3:
        continue
    if abs(float(res[0])) > 1000 or abs(float(res[1])) > 1000:
        continue
    startPos = [float(res[0]),float(res[1]),float(res[2])]
    
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    boxId = p.loadURDF("cube.urdf", startPos, startOrientation, globalScaling=0.1)
    
    time.sleep(1./240.)

"""
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
"""