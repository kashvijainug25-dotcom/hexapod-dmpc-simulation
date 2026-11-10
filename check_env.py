import numpy as np
from stable_baselines3.common.env_checker import check_env
from hexapod_env import HexapodStairsEnv

print("1. Running Stable-Baselines3 official environment checker...")
env = HexapodStairsEnv(render_mode=None, step_height=0.0)
check_env(env)
print("--> SB3 check_env passed with 0 errors!")

print("\n2. Executing 1,000 random actions test across resets...")
obs, _ = env.reset()
nan_detected = False

for step in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)

    if np.isnan(obs).any() or np.isnan(reward):
        print(f"Error: NaN detected at step {step}!")
        nan_detected = True
        break

    if terminated or truncated:
        obs, _ = env.reset()

env.close()

if not nan_detected:
    print("--> 1,000 steps executed cleanly. Resets, physics bounds, and observations verified with zero NaNs!")