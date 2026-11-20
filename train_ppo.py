import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from hexapod_env import HexapodStairsEnv


def main():
    parser = argparse.ArgumentParser(description="Train PPO baseline for Hexapod locomotion")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["smoke", "full"],
        default="smoke",
        help="smoke: 50k steps (1 env) | full: 500k steps (4 parallel envs)",
    )
    parser.add_argument("--step-height", type=float, default=0.0, help="Stair step height in meters")
    args = parser.parse_args()

    ckpt_dir = "checkpoints"
    log_dir = "tb_logs"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.stage == "smoke":
        total_timesteps = 50_000
        n_envs = 1
        vec_cls = DummyVecEnv
        run_name = "ppo_flat_smoke"
        save_freq = 25_000
    else:
        total_timesteps = 500_000
        n_envs = 4
        vec_cls = SubprocVecEnv
        run_name = "ppo_flat_parallel"
        save_freq = 50_000 // n_envs

    env = make_vec_env(
        lambda: HexapodStairsEnv(render_mode=None, step_height=args.step_height),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=vec_cls,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=ckpt_dir,
        name_prefix=run_name,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir,
        verbose=1,
        seed=42,
    )

    print(f"--- Starting {args.stage.upper()} PPO Run ({total_timesteps:,} steps, {n_envs} env(s)) ---")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=run_name,
        progress_bar=True,
    )

    final_path = os.path.join(ckpt_dir, f"{run_name}_final.zip")
    model.save(final_path)
    print(f"Training complete! Final model saved to {final_path}")
    env.close()


if __name__ == "__main__":
    main()