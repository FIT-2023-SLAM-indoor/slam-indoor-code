import pybullet as p
import time
import pybullet_data
import math


physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)
stadium = p.loadSDF("stadium.sdf")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
p.resetDebugVisualizerCamera(cameraYaw=0.0, cameraPitch=-40.0,cameraDistance=7.5,cameraTargetPosition=[0.0,0.0,0.0])



file = open("../data/video_report/test.txt",'r')
lst = file.readlines()
file.close()



for i, val in enumerate(lst):  
    res = lst[i].replace('[', '').replace(']', '').replace(';', '').split(", ")
    if len(res) != 3:
        continue
    if abs(float(res[0])) > 1000 or abs(float(res[1])) > 1000:
        continue
    startPos = [float(res[0]),float(res[1]),float(res[2]) + 3]

    startOrientation = p.getQuaternionFromEuler([0,0,0])
    boxId = p.loadURDF("cube.urdf", startPos, startOrientation, globalScaling=0.12)
    time.sleep(0.000001)

for phi in range(0, 181, 10):
    for theta in range(0, 361, 18):
        x = 2.2 * math.sin(math.radians(phi)) * math.cos(math.radians(theta)) 
        y = 2.2 * math.sin(math.radians(phi)) * math.sin(math.radians(theta))
        z = 2.2 * math.cos(math.radians(phi)) + 3
    
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        boxId = p.loadURDF("sphere2red.urdf", [x, y, z], startOrientation, globalScaling=0.12)
    
    time.sleep(0.000001)


def move_xyz(x, y, z):
    p.resetDebugVisualizerCamera(cameraYaw=cam[8], cameraPitch=cam[9],cameraDistance=cam[10],cameraTargetPosition=[x, y, z])
def move_yaw(yaw):
    p.resetDebugVisualizerCamera(cameraYaw=yaw, cameraPitch=cam[9],cameraDistance=cam[10],cameraTargetPosition=cam[11])
def move_pitch(pitch):
    p.resetDebugVisualizerCamera(cameraYaw=cam[8], cameraPitch=pitch,cameraDistance=cam[10],cameraTargetPosition=cam[11])

gui_flag = True
while(1):
    keys = p.getKeyboardEvents()
    cam = p.getDebugVisualizerCamera()

    if keys.get(p.B3G_RIGHT_ARROW):  # >
        xyz = cam[11]
        yaw = cam[8]
        x = float(xyz[0]) + math.cos(math.radians(yaw)) * 0.125
        y = float(xyz[1]) + math.sin(math.radians(yaw)) * 0.125
        move_xyz(x, y, xyz[2])
    if keys.get(p.B3G_LEFT_ARROW):  # <
        xyz = cam[11]
        yaw = cam[8]
        x = float(xyz[0]) - math.cos(math.radians(yaw)) * 0.125
        y = float(xyz[1]) - math.sin(math.radians(yaw)) * 0.125
        move_xyz(x, y, xyz[2])
    if keys.get(p.B3G_UP_ARROW):  # ^
        xyz = cam[11]
        yaw = cam[8]
        x = float(xyz[0]) - math.sin(math.radians(yaw)) * 0.125
        y = float(xyz[1]) + math.cos(math.radians(yaw)) * 0.125
        move_xyz(x, y, xyz[2])
    if keys.get(p.B3G_DOWN_ARROW):  # v 
        xyz = cam[11]
        yaw = cam[8]
        x = float(xyz[0]) + math.sin(math.radians(yaw)) * 0.125
        y = float(xyz[1]) - math.cos(math.radians(yaw)) * 0.125
        move_xyz(x, y, xyz[2])
    
    if keys.get(ord('z')):  # Roll left
        yaw = cam[8] + 1.25
        move_yaw(yaw)
    if keys.get(ord('x')):  # Roll right
        yaw = cam[8] - 1.25
        move_yaw(yaw)

    if keys.get(ord('f')):  # Roll up
        pitch = cam[9] - 1.25
        move_pitch(pitch)
    if keys.get(ord('c')):  # Roll down
        pitch = cam[9] + 1.25
        move_pitch(pitch)
    
    if keys.get(ord('q')):  # Z up
        xyz = cam[11]
        z = float(xyz[2]) + 0.125
        move_xyz(xyz[0], xyz[1], z)
    if keys.get(ord('a')):  # Z down
        xyz = cam[11]
        z = float(xyz[2]) - 0.125
        move_xyz(xyz[0], xyz[1], z)

    time.sleep(0.000001)
