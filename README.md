# 18-DOF Hexapod Stair Locomotion: Model-Free RL (PPO) vs. Diffusion MPC

Physics-grounded PyBullet & Gymnasium benchmark evaluating zero-shot stair-climbing generalization of a flat-trained model-free reinforcement learning policy (**PPO**) across procedural stair heights, establishing the baseline for **Diffusion Model Predictive Control (D-MPC)** trajectory planning and sim-to-real hardware deployment.

## Repository Architecture
- `generate_urdf.py`: Procedurally generates the 18-DOF radial hexapod (`hexapod.urdf`) with 6 legs (coxa, femur, tibia), verified inertial tensors, and torque/velocity limits (`45 N·m`, `4.0 rad/s`).
- `stairs.py`: Generates procedural multi-step staircases (`step_height` configurable from `0.00 m` to `0.03+ m`) with calibrated contact friction.
- `hexapod_env.py`: Custom Gymnasium environment (`HexapodStairsEnv`, 44-D proprioceptive state space, 18-D continuous joint offset action space) with high-stance initialization (`z ≈ 0.123 m`) and real-time camera tracking.
- `train_ppo.py`: Trains the `MlpPolicy` PPO baseline (`--stage smoke` for 50k steps; `--stage full` for 500k steps across 4 `SubprocVecEnv` parallel workers) with periodic checkpointing for offline trajectory dataset generation.
- `evaluate_policy.py`: Evaluates deterministic policies across stair heights (`n = 20` episodes per height) and supports interactive PyBullet GUI replay (`--gui`).
- `plot_results.py`: Generates 300-DPI learning curves and multi-height degradation plots from TensorBoard logs.

## Quickstart
```bash
python generate_urdf.py
python train_ppo.py --stage full
python evaluate_policy.py --model checkpoints/ppo_flat_parallel_final.zip --episodes 20
python plot_results.py
