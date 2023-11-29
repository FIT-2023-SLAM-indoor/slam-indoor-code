import pybullet as p
# import time
import pybullet_data
import math


#'''
physicsClient = p.connect(p.GUI)   # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0,0,0)
stadium = p.loadSDF("stadium.sdf")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
p.resetDebugVisualizerCamera(cameraYaw=0.0, cameraPitch=-40.0,cameraDistance=5.5,cameraTargetPosition=[0.0,-3.0,5.0])


def move_xyz(x, y, z):
    p.resetDebugVisualizerCamera(cameraYaw=cam[8], cameraPitch=cam[9],cameraDistance=cam[10],cameraTargetPosition=[x, y, z])
def move_yaw(yaw):
    p.resetDebugVisualizerCamera(cameraYaw=yaw, cameraPitch=cam[9],cameraDistance=cam[10],cameraTargetPosition=cam[11])
def move_pitch(pitch):
    p.resetDebugVisualizerCamera(cameraYaw=cam[8], cameraPitch=pitch,cameraDistance=cam[10],cameraTargetPosition=cam[11])

def visible(line):
    res = line.replace('[', '').replace(']', '').replace(';', '').split(", ")
    startPos = [float(res[0]),float(res[1]),float(res[2])]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    p.loadURDF("sphere2red.urdf", startPos, startOrientation, globalScaling=0.12)


points_data = open("../data/video_report/test.txt",'r')    
visualize_flag = True
while(True):
    if visualize_flag:
        line = points_data.readline()
        if line == '':
            visualize_flag = False
            points_data.close()
        else:
            visible(line)


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
        yaw = cam[8] - 1.25
        move_yaw(yaw)
    if keys.get(ord('x')):  # Roll right
        yaw = cam[8] + 1.25
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


''' ### Part of code for writting points cooridinates
with open("../data/video_report/test.txt",'w') as file:
    file.write("[")
    for phi in range(0, 181, 10):
        for theta in range(0, 361, 18):
            x = 4.0 * math.sin(math.radians(phi)) * math.cos(math.radians(theta)) 
            y = 4.0 * math.sin(math.radians(phi)) * math.sin(math.radians(theta))
            z = 4.0 * math.cos(math.radians(phi)) + 4.1
            file.write(f";\n {x}, {y}, {z}")
    file.write("]")
'''#