import math


def generate_hexapod_urdf(filename="hexapod.urdf"):
    # 6 legs arranged radially every 60 degrees around the central cylinder
    # Angles in radians: 30, 90, 150, 210, 270, 330 deg
    leg_angles_deg = [30, 90, 150, 210, 270, 330]
    body_radius = 0.14
    coxa_len = 0.04
    femur_len = 0.08
    tibia_len = 0.12

    urdf = [
        '<?xml version="1.0"?>',
        '<robot name="hexapod">',
        '  <link name="base_link">',
        '    <inertial>',
        '      <mass value="1.8"/>',
        '      <inertia ixx="0.008" ixy="0" ixz="0" iyy="0.008" iyz="0" izz="0.015"/>',
        '    </inertial>',
        '    <visual>',
        f'      <geometry><cylinder radius="{body_radius}" length="0.04"/></geometry>',
        '      <material name="blue"><color rgba="0.1 0.4 0.8 1.0"/></material>',
        '    </visual>',
        '    <collision>',
        f'      <geometry><cylinder radius="{body_radius}" length="0.04"/></geometry>',
        '    </collision>',
        '  </link>',
    ]

    for idx, deg in enumerate(leg_angles_deg, start=1):
        rad = math.radians(deg)
        ox = body_radius * math.cos(rad)
        oy = body_radius * math.sin(rad)

        urdf.extend([
            f'  <!-- LEG {idx} ({deg} deg) -->',
            f'  <link name="coxa_link_{idx}">',
            '    <inertial><mass value="0.08"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>',
            '    <visual><geometry><box size="0.05 0.03 0.03"/></geometry><material name="gray"><color rgba="0.6 0.6 0.6 1"/></material></visual>',
            '    <collision><geometry><box size="0.05 0.03 0.03"/></geometry></collision>',
            '  </link>',
            f'  <joint name="leg_{idx}_coxa_joint" type="revolute">',
            f'    <parent link="base_link"/><child link="coxa_link_{idx}"/>',
            f'    <origin xyz="{ox:.4f} {oy:.4f} 0.0" rpy="0 0 {rad:.4f}"/>',
            '    <axis xyz="0 0 1"/>',
            '    <limit lower="-1.2" upper="1.2" effort="45.0" velocity="4.0"/>',
            '  </joint>',
            f'  <link name="femur_link_{idx}">',
            '    <inertial><mass value="0.12"/><inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/></inertial>',
            f'    <visual><origin xyz="{femur_len/2:.3f} 0 0"/><geometry><box size="{femur_len} 0.025 0.025"/></geometry><material name="dark"><color rgba="0.2 0.2 0.2 1"/></material></visual>',
            f'    <collision><origin xyz="{femur_len/2:.3f} 0 0"/><geometry><box size="{femur_len} 0.025 0.025"/></geometry></collision>',
            '  </link>',
            f'  <joint name="leg_{idx}_femur_joint" type="revolute">',
            f'    <parent link="coxa_link_{idx}"/><child link="femur_link_{idx}"/>',
            f'    <origin xyz="{coxa_len} 0 0" rpy="0 0 0"/>',
            '    <axis xyz="0 1 0"/>',
            '    <limit lower="-1.5" upper="1.5" effort="45.0" velocity="4.0"/>',
            '  </joint>',
            f'  <link name="tibia_link_{idx}">',
            '    <inertial><mass value="0.08"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>',
            f'    <visual><origin xyz="{tibia_len/2:.3f} 0 0"/><geometry><box size="{tibia_len} 0.02 0.02"/></geometry><material name="gray"/></visual>',
            f'    <collision><origin xyz="{tibia_len/2:.3f} 0 0"/><geometry><box size="{tibia_len} 0.02 0.02"/></geometry></collision>',
            '  </link>',
            f'  <joint name="leg_{idx}_tibia_joint" type="revolute">',
            f'    <parent link="femur_link_{idx}"/><child link="tibia_link_{idx}"/>',
            f'    <origin xyz="{femur_len} 0 0" rpy="0 0 0"/>',
            '    <axis xyz="0 1 0"/>',
            '    <limit lower="-2.0" upper="1.0" effort="45.0" velocity="4.0"/>',
            '  </joint>',
        ])

    urdf.append("</robot>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(urdf))
    print(f"Generated {filename} successfully (18 revolute joints).")


if __name__ == "__main__":
    generate_hexapod_urdf()