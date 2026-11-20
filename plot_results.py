import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tb_scalars(tb_root="tb_logs", tag="rollout/ep_rew_mean"):
    event_files = glob.glob(os.path.join(tb_root, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in '{tb_root}/'.")

    # Pick the largest/latest event file (the 500k parallel run)
    latest_event = max(event_files, key=os.path.getsize)
    print(f"Reading TensorBoard logs from: {latest_event}")

    ea = EventAccumulator(latest_event)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    if tag not in available_tags:
        raise KeyError(f"Tag '{tag}' not found. Available scalar tags: {available_tags}")

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def plot_learning_curve(steps, rewards, out_path="results/ppo_learning_curve.png"):
    plt.figure(figsize=(8.5, 4.8), dpi=300)
    plt.plot(steps / 1000.0, rewards, color="#1f77b4", linewidth=2.2, label="PPO (4 Parallel Envs)")

    # Exponential moving average for smooth trend line
    if len(rewards) > 5:
        alpha = 0.25
        ema = np.zeros_like(rewards)
        ema[0] = rewards[0]
        for i in range(1, len(rewards)):
            ema[i] = alpha * rewards[i] + (1 - alpha) * ema[i - 1]
        plt.plot(steps / 1000.0, ema, color="#d62728", linestyle="--", linewidth=1.8, label="Smoothed Trend (EMA)")

    plt.title("PPO Locomotion Training Curve (Flat Ground, 18-DOF Hexapod)", fontsize=12, fontweight="bold")
    plt.xlabel("Environment Timesteps (Thousands)", fontsize=11)
    plt.ylabel("Mean Episode Reward", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_stair_degradation(out_path="results/ppo_stair_degradation.png"):
    # Verified 20-episode benchmark results from evaluate_policy.py
    step_heights_cm = ["0.0 cm\n(Flat)", "1.0 cm", "2.0 cm", "3.0 cm"]
    mean_dists = [4.562, 2.802, 1.913, 0.899]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    plt.figure(figsize=(8.0, 4.8), dpi=300)
    bars = plt.bar(step_heights_cm, mean_dists, color=colors, width=0.52, edgecolor="black", linewidth=0.9)

    for bar, dist in zip(bars, mean_dists):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.12,
            f"{dist:.3f} m\n(0/20 falls)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.ylim(0, 5.4)
    plt.title("Zero-Shot Stair Generalization of Flat-Trained PPO Baseline (n = 20 / height)", fontsize=11.5, fontweight="bold")
    plt.xlabel("Procedural Stair Step Height", fontsize=11)
    plt.ylabel("Mean Forward Displacement (m)", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    os.makedirs("results", exist_ok=True)
    steps, rewards = load_tb_scalars(tb_root="tb_logs", tag="rollout/ep_rew_mean")
    plot_learning_curve(steps, rewards)
    plot_stair_degradation()


if __name__ == "__main__":
    main()