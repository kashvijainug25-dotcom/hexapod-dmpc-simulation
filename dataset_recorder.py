import os
import numpy as np
from hexapod_env import HexapodStairsEnv

class EpisodeRecorder:
    def __init__(self, save_dir="dataset_logs"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.reset_buffer()

    def reset_buffer(self):
        self.observations = []
        self.actions = []
        self.next_observations = []
        self.rewards = []
        self.dones = []
        self.step_heights = []

    def record_step(self, obs, action, next_obs, reward, done, step_height):
        self.observations.append(obs)
        self.actions.append(action)
        self.next_observations.append(next_obs)
        self.rewards.append(reward)
        self.dones.append(done)
        self.step_heights.append(step_height)

    def save_batch(self, batch_index=0):
        filename = os.path.join(self.save_dir, f"transitions_batch_{batch_index:03d}.npz")
        np.savez_compressed(
            filename,
            observations=np.array(self.observations, dtype=np.float32),
            actions=np.array(self.actions, dtype=np.float32),
            next_observations=np.array(self.next_observations, dtype=np.float32),
            rewards=np.array(self.rewards, dtype=np.float32),
            dones=np.array(self.dones, dtype=np.bool_),
            step_heights=np.array(self.step_heights, dtype=np.float32)
        )
        print(f"Saved batch archive: {filename} ({len(self.observations)} transitions)")
        self.reset_buffer()
        return filename

if __name__ == "__main__":
    print("--- Recording 5 Episodes of Random Exploration Data ---")
    stair_height = 0.02
    env = HexapodStairsEnv(render_mode=None, step_height=stair_height)
    recorder = EpisodeRecorder()

    total_episodes = 5
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            recorder.record_step(obs, action, next_obs, reward, done, stair_height)
            obs = next_obs

    saved_path = recorder.save_batch(batch_index=1)
    env.close()

    # Load file back and verify array shapes
    print("\n--- Verifying Compressed Archive Shapes ---")
    data = np.load(saved_path)
    for key in data.files:
        print(f"Field: {key:18s} | Shape: {data[key].shape} | Dtype: {data[key].dtype}")
    print("\nDataset logging verified successfully!")
    