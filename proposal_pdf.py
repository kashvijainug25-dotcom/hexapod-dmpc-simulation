import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


def wrap(text, width=106):
    return "\n".join(textwrap.wrap(text.strip(), width=width))


def build_one_page_proposal(
    plot_path="results/ppo_stair_degradation.png",
    out_pdf="Kashvi_Jain_Caltech_SURF_2027_Hexapod_DMPC_Proposal.pdf",
):
    fig = plt.figure(figsize=(8.5, 11.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header Block
    ax.text(
        0.06,
        0.955,
        "Caltech SURF 2027 Research Proposal Summary",
        fontsize=9.5,
        fontweight="bold",
        color="#1f4e78",
    )
    ax.text(
        0.06,
        0.932,
        "Horizon-Aware Legged Locomotion over Unstructured Stairs: Benchmarking\n"
        "Diffusion Model Predictive Control (D-MPC) vs. Model-Free RL on an 18-DOF Hexapod",
        fontsize=12.0,
        fontweight="bold",
        color="#111111",
        va="top",
    )
    ax.text(
        0.06,
        0.878,
        "Prospective Fellow: Ms. Kashvi Jain  |  B.Tech. Mechanical Engineering (Class of 2028), NSUT Delhi\n"
        "Code & Benchmark Repository: https://github.com/kashvijainug25-dotcom/hexapod-dmpc-simulation",
        fontsize=8.6,
        color="#333333",
        va="top",
    )
    ax.plot([0.06, 0.94], [0.842, 0.842], color="#1f4e78", linewidth=1.2)

    # Section 1: Problem Statement & Motivation
    ax.text(0.06, 0.824, "1. Problem Statement & Motivation", fontsize=10.5, fontweight="bold", color="#1f4e78")
    sec1 = wrap(
        "Reactive, model-free reinforcement learning (RL) policies achieve high locomotion velocity on flat ground "
        "but struggle to generalize zero-shot to abrupt geometric discontinuities such as stairs, curbs, and rubble. "
        "Without multi-step horizon lookahead, reactive controllers fail to modulate swing-leg clearance prior to "
        "riser impact. Conversely, Diffusion Model Predictive Control (D-MPC) frames trajectory optimization as "
        "conditional denoising over multimodal state-action sequences, enabling warm-started, gradient-free "
        "receding-horizon planning across contact-rich transitions. This project tests whether D-MPC outperforms "
        "model-free RL (Proximal Policy Optimization) in zero-shot stair traversal and quantifies the resulting "
        "sim-to-real transfer gap.",
        width=112,
    )
    ax.text(0.06, 0.806, sec1, fontsize=8.5, color="#1a1a1a", va="top", linespacing=1.32)

    # Section 2: Preliminary Benchmark & Figure
    ax.text(
        0.06,
        0.678,
        "2. Preliminary Baseline Completed (18-DOF PyBullet & Gymnasium Benchmark)",
        fontsize=10.5,
        fontweight="bold",
        color="#1f4e78",
    )
    sec2 = wrap(
        "Ahead of SURF, I built a reproducible 18-DOF radial hexapod simulation (45 N·m peak joint torque, "
        "4.0 rad/s velocity limit, 0.123 m nominal chassis clearance) with procedural staircases (0.00–0.03 m risers). "
        "A 500k-step parallel PPO baseline trained on flat ground was evaluated zero-shot across 4 stair step heights "
        "(n = 20 trials/height). While PPO maintained postural stability (0/20 falls), forward traversal degraded "
        "monotonically by 80.3% (from 4.562 m to 0.899 m) due to reactive toe-riser collisions.",
        width=112,
    )
    ax.text(0.06, 0.660, sec2, fontsize=8.5, color="#1a1a1a", va="top", linespacing=1.32)

    # Embedded Benchmark Table (Left)
    table_data = [
        ["Step Height", "Mean Dist (m)", "Success", "Falls"],
        ["0.000 m (Flat)", "4.562 ± 0.000", "100.0%", "0 / 20"],
        ["0.010 m (1 cm)", "2.802 ± 0.000", "100.0%", "0 / 20"],
        ["0.020 m (2 cm)", "1.913 ± 0.000", "100.0%", "0 / 20"],
        ["0.030 m (3 cm)", "0.899 ± 0.000", "100.0%", "0 / 20"],
    ]
    tbl_ax = fig.add_axes([0.06, 0.395, 0.42, 0.175])
    tbl_ax.axis("off")
    table = tbl_ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
        colWidths=[0.31, 0.31, 0.19, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1.0, 1.38)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if row == 0:
            cell.set_facecolor("#1f4e78")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f7fa")

    # Embedded Degradation Plot (Right)
    if os.path.exists(plot_path):
        img_ax = fig.add_axes([0.50, 0.382, 0.44, 0.205])
        img = mpimg.imread(plot_path)
        img_ax.imshow(img)
        img_ax.axis("off")
    else:
        ax.text(0.52, 0.48, "[Run plot_results.py first to embed figure]", fontsize=8.5, color="red")

    # Section 3: 10-Week SURF 2027 Research Plan
    ax.text(0.06, 0.365, "3. 10-Week SURF 2027 Research Plan", fontsize=10.5, fontweight="bold", color="#1f4e78")
    plan_items = [
        "• Weeks 1–2 (Offline Trajectory Dataset & Terrain Sensing): Augment the 44-D proprioceptive state with local "
        "terrain elevation queries and collect mixed-quality trajectories D = {(s_t, a_t)} across 50k–500k PPO "
        "checkpoints and heuristic tripod/wave gaits on randomized stair geometries.",
        "• Weeks 3–5 (Conditional Diffusion Trajectory Model): Train a 1-D U-Net / Transformer denoising diffusion "
        "planner p_theta(tau_{0:H} | s_0, g) predicting horizon-H joint-state and action trajectories conditioned on "
        "current proprioception, local terrain profile, and target forward velocity.",
        "• Weeks 6–7 (Receding-Horizon D-MPC): Implement closed-loop D-MPC by sampling K candidate trajectory plans "
        "per control cycle, scoring rollouts on forward progress, foot-riser clearance, and energy cost, and executing "
        "the top sequence with warm-started reverse diffusion.",
        "• Weeks 8–10 (Sim-to-Real Sensitivity & Hardware Validation): Quantify degradation under injected actuator "
        "latency (10–30 ms), joint friction/mass shifts (±20%), and sensor noise across PPO and D-MPC, followed by "
        "physical 18-DOF hexapod testbed validation and the SURF final paper.",
    ]
    y_cursor = 0.346
    for item in plan_items:
        wrapped = wrap(item, width=112)
        ax.text(0.06, y_cursor, wrapped, fontsize=8.3, color="#1a1a1a", va="top", linespacing=1.28)
        y_cursor -= 0.0155 * (wrapped.count("\n") + 1) + 0.008

    # Section 4: Expected Outcomes
    ax.text(0.06, y_cursor - 0.006, "4. Expected Outcomes", fontsize=10.5, fontweight="bold", color="#1f4e78")
    sec4 = wrap(
        "(1) Controlled quantitative benchmark comparing zero-shot stair traversal (1.0–4.0 cm risers), sample "
        "efficiency, and riser-recovery behavior between PPO and D-MPC; (2) Open-source benchmark suite and "
        "sim-to-real sensitivity characterization for contact-rich legged locomotion.",
        width=112,
    )
    ax.text(0.06, y_cursor - 0.024, sec4, fontsize=8.4, color="#1a1a1a", va="top", linespacing=1.30)

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Generated 1-page proposal PDF: {out_pdf}")


if __name__ == "__main__":
    build_one_page_proposal()