import math

def build_hexapod_urdf(filename="hexapod.urdf"):
    # 6 legs radially positioned around the base at 60-degree increments
    leg_angles = [30, 90, 150, 210, 270, 330]
    radius = 0.14  # Distance from base origin to leg attachment (meters)

    urdf = []
    urdf.append('<?xml version="1.0"?>')
    urdf.append('<robot name="hexapod">')

    # Main Body Link
    urdf.append('  <link name="base_link">')
    urdf.append('    <inertial>')
    urdf.append('      <mass value="1.8"/>')
    urdf.append('      <inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.015"/>')
    urdf.append('    </inertial>')
    urdf.append('    <visual>')
    urdf.append('      <geometry><cylinder radius="0.14" length="0.04"/></geometry>')
    urdf.append('      <material name="blue"><color rgba="0.1 0.4 0.8 1.0"/></material>')
    urdf.append('    </visual>')
    urdf.append('    <collision>')
    urdf.append('      <geometry><cylinder radius="0.14" length="0.04"/></geometry>')
    urdf.append('    </collision>')
    urdf.append('  </link>')

    # 6 Legs: each having 3 revolute joints (coxa, femur, tibia)
    for i, deg in enumerate(leg_angles):
        leg = i + 1
        rad = math.radians(deg)
        bx = radius * math.cos(rad)
        by = radius * math.sin(rad)

        # 1. Coxa Link & Joint (Yaw / Abduction)
        urdf.append(f'  <link name="coxa_link_{leg}">')
        urdf.append('    <inertial><mass value="0.08"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>')
        urdf.append('    <visual><geometry><box size="0.05 0.03 0.03"/></geometry><material name="gray"><color rgba="0.6 0.6 0.6 1"/></material></visual>')
        urdf.append('    <collision><geometry><box size="0.05 0.03 0.03"/></geometry></collision>')
        urdf.append('  </link>')
        urdf.append(f'  <joint name="leg_{leg}_coxa_joint" type="revolute">')
        urdf.append(f'    <parent link="base_link"/><child link="coxa_link_{leg}"/>')
        urdf.append(f'    <origin xyz="{bx:.4f} {by:.4f} 0.0" rpy="0 0 {rad:.4f}"/>')
        urdf.append('    <axis xyz="0 0 1"/>')
        urdf.append('    <limit lower="-1.2" upper="1.2" effort="40.0" velocity="4.0"/>')
        urdf.append('  </joint>')

        # 2. Femur Link & Joint (Pitch / Thigh lift)
        urdf.append(f'  <link name="femur_link_{leg}">')
        urdf.append('    <inertial><mass value="0.12"/><inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/></inertial>')
        urdf.append('    <visual><origin xyz="0.04 0 0"/><geometry><box size="0.08 0.025 0.025"/></geometry><material name="dark"><color rgba="0.2 0.2 0.2 1"/></material></visual>')
        urdf.append('    <collision><origin xyz="0.04 0 0"/><geometry><box size="0.08 0.025 0.025"/></geometry></collision>')
        urdf.append('  </link>')
        urdf.append(f'  <joint name="leg_{leg}_femur_joint" type="revolute">')
        urdf.append(f'    <parent link="coxa_link_{leg}"/><child link="femur_link_{leg}"/>')
        urdf.append('    <origin xyz="0.04 0 0" rpy="0 0 0"/>')
        urdf.append('    <axis xyz="0 1 0"/>')
        urdf.append('    <limit lower="-1.5" upper="1.5" effort="40.0" velocity="4.0"/>')
        urdf.append('  </joint>')

        # 3. Tibia Link & Joint (Pitch / Knee plant)
        urdf.append(f'  <link name="tibia_link_{leg}">')
        urdf.append('    <inertial><mass value="0.08"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>')
        urdf.append('    <visual><origin xyz="0.06 0 0"/><geometry><box size="0.12 0.02 0.02"/></geometry><material name="gray"/></visual>')
        urdf.append('    <collision><origin xyz="0.06 0 0"/><geometry><box size="0.12 0.02 0.02"/></geometry></collision>')
        urdf.append('  </link>')
        urdf.append(f'  <joint name="leg_{leg}_tibia_joint" type="revolute">')
        urdf.append(f'    <parent link="femur_link_{leg}"/><child link="tibia_link_{leg}"/>')
        urdf.append('    <origin xyz="0.08 0 0" rpy="0 0 0"/>')
        urdf.append('    <axis xyz="0 1 0"/>')
        urdf.append('    <limit lower="-2.0" upper="1.0" effort="40.0" velocity="4.0"/>')
        urdf.append('  </joint>')

    urdf.append('</robot>')

    with open(filename, "w") as f:
        f.write("\n".join(urdf))
    print(f"Generated {filename} successfully with 18 revolute joints!")

if __name__ == "__main__":
    build_hexapod_urdf()