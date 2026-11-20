import os
import glob
import time
import argparse
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from hexapod_env import HexapodStairsEnv


def find_latest_checkpoint(ckpt_dir="checkpoints"):
    zips = glob.glob(os.path.join(ckpt_dir, "*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip model files found in '{ckpt_dir}/'.")
    return max(zips, key=os.path.getmtime)


def evaluate_at_height(model, step_height, n_episodes=20, render=False, success_dist=0.5):
    render_mode = "human" if render else None
    env = HexapodStairsEnv(render_mode=render_mode, step_height=step_height)

    distances = []
    falls = 0
    successes = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=100 + ep)
        start_x = float(info.get("x_pos", 0.0))
        final_x = start_x
        fell = False
        done = False

        if render and ep == 0:
            time.sleep(1.5)  # Brief pause so you can hit Win + Alt + R before walking starts

        while not done:
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                final_x = float(info.get("x_pos", final_x))

                if render:
                    time.sleep(1.0 / 60.0)

                if terminated:
                    fell = True
                done = terminated or truncated
            except p.error:
                done = True
                break

        dist = final_x - start_x
        distances.append(dist)
        if fell:
            falls += 1
        if not fell and dist >= success_dist:
            successes += 1

    env.close()
    return {
        "step_height": step_height,
        "mean_dist": float(np.mean(distances)),
        "std_dist": float(np.std(distances)),
        "success_rate": 100.0 * successes / n_episodes,
        "fall_rate": 100.0 * falls / n_episodes,
        "falls": falls,
        "n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO Hexapod Policy across stair heights")
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint .zip")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes per step height")
    parser.add_argument("--heights", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.03])
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI for visual inspection")
    args = parser.parse_args()

    model_path = args.model if args.model else find_latest_checkpoint()
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    if args.gui:
        h = args.heights[0]
        print(f"\n--- GUI Visual Clip Mode (1 episode at step_height={h} m) ---")
        print("Press Win + Alt + R as soon as the PyBullet window opens!")
        res = evaluate_at_height(model, step_height=h, n_episodes=1, render=True)
        print(f"GUI Run -> Distance: {res['mean_dist']:.3f} m | Falls: {res['falls']}/1")
        return

    print(f"\nEvaluating across {len(args.heights)} step heights ({args.episodes} episodes each)...\n")
    print(f"{'Step Height (m)':<18} | {'Mean Dist (m)':<18} | {'Success Rate':<14} | {'Falls'}")
    print("-" * 68)

    for h in args.heights:
        res = evaluate_at_height(model, step_height=h, n_episodes=args.episodes, render=False)
        print(
            f"{h:<18.3f} | {res['mean_dist']:>6.3f} ± {res['std_dist']:<6.3f}   | "
            f"{res['success_rate']:>10.1f}%   | {res['falls']}/{res['n_episodes']}"
        )


if __name__ == "__main__":
    main()