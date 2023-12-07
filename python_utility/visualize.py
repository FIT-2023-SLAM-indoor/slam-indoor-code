import config as cfg
import pybullet as p
import pybullet_data
import math
import time


YAW_IDX = 8
PITCH_IDX = 9
DIST_IDX = 10
TARG_POS = 11
CAM_DISTANCE = 1.0
COEFF_MODES = [0.03, 0.08, 0.13, 0.25, 0.6, 1.5]


#'''
physicsClient = p.connect(p.GUI)   # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0,0,0)
p.loadURDF("plane.urdf", globalScaling=25.0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)
p.resetDebugVisualizerCamera(cameraYaw=0.0,
                             cameraPitch=-40.0,
                             cameraDistance=CAM_DISTANCE,
                             cameraTargetPosition=[0.0,-7.0,9.0])


def move_xyz(x, y, z):
    p.resetDebugVisualizerCamera(cameraYaw=cam[8], cameraPitch=cam[9],cameraDistance=cam[10],cameraTargetPosition=[x, y, z])
def move_yaw(yaw):
    p.resetDebugVisualizerCamera(cameraYaw=yaw, cameraPitch=cam[9],cameraDistance=cam[10],cameraTargetPosition=cam[11])
def move_pitch(pitch):
    p.resetDebugVisualizerCamera(cameraYaw=cam[8], cameraPitch=pitch,cameraDistance=cam[10],cameraTargetPosition=cam[11])

def visualizePointFromString(line):
    coords = line.replace('[', '').replace(']', '').replace(';', '').split(", ")
    if len(coords) == 3:
        startPos = [float(coords[0]) * cfg.COORD_X_SCALE, 
                    float(coords[2]) * cfg.COORD_Y_SCALE,
                    float(coords[1]) * cfg.COORD_Z_SCALE]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        p.loadURDF("sphere2red.urdf", startPos, startOrientation, globalScaling=0.4)
        print(lineIndex)


points_data = open(cfg.FILE_PATH,'r')
lines = points_data.readlines()
lineIndex = 0
visualize_flag = True
curr_coeff = 2
while(True):
    if visualize_flag:
        line = lines[lineIndex]
        lineIndex += 1
        if line == "FINISH":
            visualize_flag = False
            points_data.close()
            print("#######\n#######\n#######")
            print("-FILE ENDED-")
            print("#######\n#######\n#######")
        else:
            visualizePointFromString(line)

    
    keys = p.getKeyboardEvents()
    cam = p.getDebugVisualizerCamera()

    if keys.get(ord('p')): # once stops visualization
        visualize_flag = not visualize_flag

    if keys.get(p.B3G_SHIFT):
        if curr_coeff < len(COEFF_MODES) - 1:
            curr_coeff += 1
    if keys.get(p.B3G_CONTROL):
        if curr_coeff > 0:
            curr_coeff -= 1

    if keys.get(ord('c')):  # >
        xyz = cam[11]
        yaw = cam[8]
        x = float(xyz[0]) + math.cos(math.radians(yaw)) * COEFF_MODES[curr_coeff]
        y = float(xyz[1]) + math.sin(math.radians(yaw)) * COEFF_MODES[curr_coeff]
        move_xyz(x, y, xyz[2])
    if keys.get(ord('z')):  # <
        xyz = cam[11]
        yaw = cam[8]
        x = float(xyz[0]) - math.cos(math.radians(yaw)) * COEFF_MODES[curr_coeff]
        y = float(xyz[1]) - math.sin(math.radians(yaw)) * COEFF_MODES[curr_coeff]
        move_xyz(x, y, xyz[2])
    if keys.get(ord('s')):  # ^
        xyz = cam[11]
        yaw = cam[8]
        x = float(xyz[0]) - math.sin(math.radians(yaw)) * COEFF_MODES[curr_coeff]
        y = float(xyz[1]) + math.cos(math.radians(yaw)) * COEFF_MODES[curr_coeff]
        move_xyz(x, y, xyz[2])
    if keys.get(ord('x')):  # v 
        xyz = cam[11]
        yaw = cam[8]
        x = float(xyz[0]) + math.sin(math.radians(yaw)) * COEFF_MODES[curr_coeff]
        y = float(xyz[1]) - math.cos(math.radians(yaw)) * COEFF_MODES[curr_coeff]
        move_xyz(x, y, xyz[2])
    
    if keys.get(p.B3G_LEFT_ARROW):  # Left yaw
        yaw = cam[8] - COEFF_MODES[curr_coeff] * 5
        move_yaw(yaw)
    if keys.get(p.B3G_RIGHT_ARROW):  # Right yaw
        yaw = cam[8] + COEFF_MODES[curr_coeff] * 5
        move_yaw(yaw)

    if keys.get(p.B3G_DOWN_ARROW):  # Up pitch
        pitch = cam[9] - COEFF_MODES[curr_coeff] * 5
        move_pitch(pitch)
    if keys.get(p.B3G_UP_ARROW):  # Down pithc
        pitch = cam[9] + COEFF_MODES[curr_coeff] * 5
        move_pitch(pitch)
    
    if keys.get(ord('q')):  # Z up
        xyz = cam[11]
        z = float(xyz[2]) + COEFF_MODES[curr_coeff] * 5
        move_xyz(xyz[0], xyz[1], z)
    if keys.get(ord('a')):  # Z down
        xyz = cam[11]
        z = float(xyz[2]) - COEFF_MODES[curr_coeff] * 5
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