import pybullet as p


def build_stairs(
    num_steps=8,
    step_height=0.02,
    step_depth=0.15,
    step_width=0.8,
    start_x=0.35,
    physics_client=0,
):
    """Builds a staircase of static box steps starting at start_x along +X."""
    if step_height <= 0.0:
        return []

    stair_ids = []
    for i in range(num_steps):
        h = (i + 1) * step_height
        half_extents = [step_depth / 2.0, step_width / 2.0, h / 2.0]
        pos_x = start_x + (i + 0.5) * step_depth
        pos_z = h / 2.0

        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=physics_client
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.75, 0.55, 0.35, 1.0],
            physicsClientId=physics_client,
        )
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[pos_x, 0.0, pos_z],
            physicsClientId=physics_client,
        )
        p.changeDynamics(
            body_id, -1, lateralFriction=1.0, physicsClientId=physics_client
        )
        stair_ids.append(body_id)

    return stair_ids