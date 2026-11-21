# 18-DOF Hexapod Stair Locomotion: Model-Free RL (PPO) vs. Diffusion MPC

Physics-grounded PyBullet & Gymnasium simulation framework evaluating zero-shot stair-climbing generalization of a flat-trained model-free reinforcement learning policy (**PPO**) across procedural stair heights (`0.00 m` to `0.03 m`), establishing the quantitative baseline for **Diffusion Model Predictive Control (D-MPC)** trajectory planning and sim-to-real deployment.

---

## Development Progression (Days 1–3)

### Day 1 — Robot Modeling, Kinematics & High-Stance Calibration
* **18-DOF Radial URDF (`generate_urdf.py` → `hexapod.urdf`):** Procedurally generated a 6-legged radial hexapod (`30°, 90°, 150°, 210°, 270°, 330°` attachment angles) with 3 revolute joints per leg (`coxa`, `femur`, `tibia`), calibrated link masses (`1.8 kg` chassis, `0.08–0.12 kg` leg segments), diagonal inertia tensors, and actuator limits (`45.0 N·m` peak torque, `4.0 rad/s` velocity cap).
* **Kinematics & High-Stance Standing Posture:** Resolved right-hand-rule $+Y$ joint polarity (`stand_coxa = 0.0 rad`, `stand_femur = +45.0°`, `stand_tibia = -15.0°`) with active spawn holding torque, maintaining a steady settled chassis clearance of **`z ≈ 0.123 m`** above ground to prevent early knee-contact resets and clear stair risers.

### Day 2 — Procedural Terrain & Custom Gymnasium Environment
* **Procedural Staircase Generator (`stairs.py`):** Built parameterized multi-step staircases (`8` steps, `0.15 m` tread depth, `0.80 m` width, configurable `step_height` from `0.00 m` to `0.03+ m`) with `lateralFriction = 1.0`.
* **Gymnasium Locomotion Environment (`hexapod_env.py`):** Implemented `HexapodStairsEnv` with a **44-D continuous observation space** (18 joint positions, 18 joint velocities, roll, pitch, 3-D angular velocity, 3-D linear velocity) and an **18-D continuous action space** (`[-1, 1]` scaled joint angle offsets around the nominal standing pose), verified with zero `NaN`s via Stable-Baselines3 `check_env`.
* **Shaped Locomotion Reward:** Combined forward velocity tracking (`+2.0 * v_x`), chassis height clearance (`+0.5 * max(0, z - 0.10)`), roll/pitch stabilization (`-0.5 * (|roll| + |pitch|)`), and control effort regularization (`-0.01 * ||a||²`), with fall termination at `z < 0.07 m` or tilt `> 0.8 rad`.

### Day 3 — Parallel PPO Training, Zero-Shot Stair Benchmark & Visualization
* **Parallel Baseline Training (`train_ppo.py`):** Trained an `MlpPolicy` PPO agent on flat ground (`step_height = 0.0 m`) using 4 parallel `SubprocVecEnv` workers (`500,000` timesteps, `lr = 3e-4`, `n_steps = 2048`, `batch_size = 64`, `gamma = 0.99`, `gae_lambda = 0.95`), saving periodic checkpoints (`50k`–`500k`) for downstream offline trajectory dataset generation.
* **Multi-Height Stair Evaluation (`evaluate_policy.py`):** Benchmarked the deterministic policy across 4 stair step heights (`20` episodes per height) and added real-time camera-tracking GUI replay (`--gui`).
* **Automated Figure Generation (`plot_results.py`):** Extracted TensorBoard training scalars (`rollout/ep_rew_mean`) and multi-height evaluation metrics into 300-DPI publication figures (`results/`).

---

## Repository Structure

```text
hexapod-dmpc-simulation/
├── generate_urdf.py                  # Procedural 18-DOF hexapod URDF builder
├── hexapod.urdf                      # Generated robot description (18 revolute joints)
├── stairs.py                         # Procedural multi-step staircase generator
├── hexapod_env.py                    # Custom Gymnasium environment (44-D obs, 18-D action)
├── train_ppo.py                      # Smoke (50k) & parallel (500k) PPO training pipeline
├── evaluate_policy.py                # Multi-height evaluation & camera-tracking GUI replay
├── plot_results.py                   # 300-DPI learning curve & degradation chart generator
├── checkpoints/
│   └── ppo_flat_parallel_final.zip   # Trained 500k-step flat-ground PPO baseline weights
└── results/
    ├── ppo_learning_curve.png        # Episode reward convergence over 500k timesteps
    └── ppo_stair_degradation.png     # Zero-shot traversal distance across stair heights