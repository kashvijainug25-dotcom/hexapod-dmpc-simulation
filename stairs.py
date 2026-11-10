import time
import pybullet as p
import pybullet_data

def build_stairs(num_steps=6, step_height=0.03, step_depth=0.15, step_width=0.8, start_x=0.4):
    """
    Builds a procedural staircase along the +X direction using mass-zero static bodies.
    step_height: height of each individual step riser (m)
    step_depth: tread length along X (m)
    step_width: width along Y (m)
    start_x: distance from origin where the first step begins (m)
    """
    stair_ids = []
    half_w = step_width / 2.0
    half_d = step_depth / 2.0

    for i in range(num_steps):
        # Total height of the step box so it rests solidly from the ground (Z=0)
        current_h = (i + 1) * step_height
        half_h = current_h / 2.0

        center_x = start_x + (i * step_depth) + half_d
        center_y = 0.0
        center_z = half_h  # Box center is at half its total height

        col_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_d, half_w, half_h]
        )
        
        # Alternating shade for visual depth
        shade = 0.7 if i % 2 == 0 else 0.55
        vis_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_d, half_w, half_h],
            rgbaColor=[shade, shade, shade + 0.05, 1.0]
        )

        body_id = p.createMultiBody(
            baseMass=0,  # Mass = 0 makes it completely static/immovable
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[center_x, center_y, center_z]
        )
        # Set high friction for realistic foothold traction
        p.changeDynamics(body_id, -1, lateralFriction=1.0)
        stair_ids.append(body_id)

    return stair_ids

if __name__ == "__main__":
    # Test script: verify robot stands right in front of procedural stairs
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=60,
        cameraPitch=-30,
        cameraTargetPosition=[0.4, 0, 0.15]
    )

    plane_id = p.loadURDF("plane.urdf")
    stairs = build_stairs(num_steps=6, step_height=0.03, step_depth=0.15, step_width=0.8, start_x=0.35)
    robot_id = p.loadURDF("hexapod.urdf", [0, 0, 0.22])

    print(f"Spawned {len(stairs)} stairs successfully. Previewing in GUI...")
    for _ in range(240 * 5):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()