import pybullet as p
import pybullet_data
import time

# Создание окна симуляции
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Загрузка моделей
planeId = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.81)

# Чтение данных из файла
with open("../../data/video_report/3Dpoints.txt", 'r') as file:
    lines = file.readlines()


# Преобразование данных в список координат
points = []
for line in lines[1:]:
    line = line.replace('[', '').replace(']', '').replace(';', '').strip()
    point = list(map(float, line.split(', ')[:-1]))
    points.append(point)


time.sleep(1.0)

# Создание визуализированных сфер для ключевых точек
sphereRadius = 0.5
visual_shapes = []
for point in points:
    visual_shapes.append(p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 0, 0, 1], radius=sphereRadius))
    
# Создание маркеров на координатах точек
for i, point in enumerate(points):
    p.createMultiBody(baseVisualShapeIndex = visual_shapes[i], basePosition = point)
