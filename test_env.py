import time
import pybullet as p
import pybullet_data

# 1. Connect to visual GUI
client = p.connect(p.GUI)

# 2. Add PyBullet standard asset paths (plane, shapes)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Set standard Earth gravity
p.setGravity(0, 0, -9.81)

# 4. Load the ground plane
plane_id = p.loadURDF("plane.urdf")

# 5. Drop a test box
box_id = p.loadURDF("cube_small.urdf", [0, 0, 1.2])

print(f"Loaded box entity ID {box_id} with {p.getNumJoints(box_id)} joints.")

# 6. Step physics simulation loop at 240 Hz for 4 seconds
for _ in range(240 * 4):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

p.disconnect()
print("PyBullet GUI test finished successfully!")