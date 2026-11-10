import math
import time
import pybullet as p
import pybullet_data

# 1. Connect GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Position camera to frame the robot clearly
p.resetDebugVisualizerCamera(
    cameraDistance=0.75,
    cameraYaw=45,
    cameraPitch=-25,
    cameraTargetPosition=[0, 0, 0.1]
)

# 2. Load ground plane and your newly generated hexapod URDF
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("hexapod.urdf", [0, 0, 0.22])

# 3. Discover joints and map revolute joints by name
num_joints = p.getNumJoints(robot_id)
joint_map = {}

print(f"\n--- Total Joints Detected: {num_joints} ---")
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    name = info[1].decode("utf-8")
    j_type = info[2]  # 0 is p.JOINT_REVOLUTE
    if j_type == p.JOINT_REVOLUTE:
        joint_map[name] = i
        print(f"Revolute Joint [{i:02d}]: {name}")

# 4. Standing Target Angles (in radians)
STAND_COXA = 0.0                   # Neutral radial outward angle
STAND_FEMUR = math.radians(24.0)   # Thigh lifts up ~24 degrees
STAND_TIBIA = math.radians(-54.0)  # Knee bends down ~54 degrees to plant foot

MAX_FORCE = 45.0  # Joint torque limit (N*m)

fps = 240.0
dt = 1.0 / fps

print("\n--- Running Standing Controller (Holding for 20 seconds) ---")
for step in range(int(fps * 20)):
    # Command position control on all 18 joints
    for name, idx in joint_map.items():
        if "coxa" in name:
            target = STAND_COXA
        elif "femur" in name:
            target = STAND_FEMUR
        elif "tibia" in name:
            target = STAND_TIBIA
        else:
            target = 0.0

        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=MAX_FORCE,
            maxVelocity=4.0
        )

    p.stepSimulation()
    time.sleep(dt)

    # Telemetry: Print base height and roll/pitch tilt every 0.5s (120 steps)
    if step % 120 == 0:
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        roll, pitch, yaw = [math.degrees(a) for a in p.getEulerFromQuaternion(orn)]
        print(f"Step {step:04d} | Height: {pos[2]:.3f} m | Roll: {roll:+6.2f}° | Pitch: {pitch:+6.2f}°")

p.disconnect()
print("Simulation finished.")