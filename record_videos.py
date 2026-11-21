import os
import numpy as np
import pybullet as p
import imageio.v2 as imageio
from stable_baselines3 import PPO
from hexapod_env import HexapodStairsEnv


def record_clip(model, step_height, out_path, max_frames=360, width=800, height=600, fps=30):
    env = HexapodStairsEnv(render_mode=None, step_height=step_height)
    obs, info = env.reset(seed=100)

    frames = []
    done = False
    step_count = 0

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=55.0, aspect=float(width) / height, nearVal=0.05, farVal=10.0
    )

    while not done and step_count < max_frames:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        # Render every 2nd control step for smooth 30 FPS playback (~12 sec clip)
        if step_count % 2 == 0:
            base_pos, _ = p.getBasePositionAndOrientation(
                env.robot_id, physicsClientId=env.client
            )
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[float(base_pos[0]), float(base_pos[1]), 0.12],
                distance=0.85,
                yaw=45,
                pitch=-25,
                roll=0,
                upAxisIndex=2,
            )
            _, _, rgba, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=env.client,
            )
            rgb = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
            frames.append(rgb)

    final_x = float(info.get("x_pos", 0.0))
    env.close()

    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)
    print(f"Saved {out_path} ({len(frames)} frames, final x = {final_x:.3f} m)")


def main():
    os.makedirs("results", exist_ok=True)
    model = PPO.load("checkpoints/ppo_flat_parallel_final.zip")

    print("Recording Clip 1: Flat ground (0.00 m)...")
    record_clip(model, step_height=0.0, out_path="results/demo_flat_walk.mp4", max_frames=480)

    print("Recording Clip 2: 2.0 cm stairs (0.02 m)...")
    record_clip(model, step_height=0.02, out_path="results/demo_stairs_2cm.mp4", max_frames=600)


if __name__ == "__main__":
    main()