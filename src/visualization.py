import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster

from .trainer import TASK_CONFIGS, normalize_task_name

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22"]


def get_hac_order(matrix: np.ndarray) -> np.ndarray:
    jittered = matrix.T + 1e-7 * np.random.RandomState(42).randn(*matrix.T.shape)
    Z = linkage(jittered, method="ward")
    return leaves_list(Z)


def get_cluster_assignments(matrix: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    jittered = matrix.T + 1e-7 * np.random.RandomState(42).randn(*matrix.T.shape)
    Z = linkage(jittered, method="ward")
    return fcluster(Z, t=n_clusters, criterion="maxclust") - 1


def generate_task_visualizations(task: str, root_dir: str):
    """Generates the complete suite of comparative charts for a task in <task_dir>/figures/."""
    task_key = normalize_task_name(task)
    tcfg = TASK_CONFIGS[task_key]
    task_title = tcfg["display_name"]
    task_dir = os.path.join(root_dir, task_key)
    fig_dir = os.path.join(task_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    en_dir = os.path.join(task_dir, "fgate_enabled")
    dis_dir = os.path.join(task_dir, "fgate_disabled")

    en_hist_path = os.path.join(en_dir, "training_history.json")
    dis_hist_path = os.path.join(dis_dir, "training_history.json")
    if not (os.path.exists(en_hist_path) and os.path.exists(dis_hist_path)):
        print(f"Skipping plots for {task_title}: missing training_history.json in {en_dir} or {dis_dir}")
        return

    with open(en_hist_path) as f:
        en_hist = json.load(f)
    with open(dis_hist_path) as f:
        dis_hist = json.load(f)

    epochs = [h["epoch"] for h in en_hist]
    en_loss = [max(h["train_loss"], 1e-7) for h in en_hist]
    dis_loss = [max(h["train_loss"], 1e-7) for h in dis_hist]
    en_tacc = [h["train_acc"] * 100.0 for h in en_hist]
    dis_tacc = [h["train_acc"] * 100.0 for h in dis_hist]
    en_vacc = [h.get("val_acc_len_40", h.get("avg_val_acc", 0.0)) * 100.0 for h in en_hist]
    dis_vacc = [h.get("val_acc_len_40", h.get("avg_val_acc", 0.0)) * 100.0 for h in dis_hist]

    label_en = "Forget Gate Enabled ($m_t = 0, f_t = \\sigma(\\tilde{f}_t)$)"
    label_dis = "Forget Gate Disabled ($m_t = 0, f_t \\equiv 1.0$)"

    # 1. Combined Training Loss & Validation Accuracy Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=300)
    ax1.plot(epochs, en_loss, label=label_en, color="#1f77b4", lw=2.1)
    ax1.plot(epochs, dis_loss, label=label_dis, color="#d62728", ls="--", lw=2.1)
    ax1.set_title(f"Training Loss Trajectory ({task_title})", pad=10, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss (Log Scale)")
    ax1.set_yscale("log")
    ax1.grid(True, which="both")
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    ax2.plot(epochs, en_vacc, label=label_en, color="#1f77b4", lw=2.1)
    ax2.plot(epochs, dis_vacc, label=label_dis, color="#d62728", ls="--", lw=2.1)
    ax2.set_title(f"Validation Accuracy Progression ({task_title})", pad=10, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_ylim(40, 103)
    ax2.grid(True)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "training_and_validation_curves.png"), dpi=300)
    plt.close()

    # Standalone curves
    for fname, title, ylabel, s_en, s_dis, yscale, ylim, lloc in [
        ("training_loss.png", f"Training Loss ({task_title})", "Cross-Entropy Loss", en_loss, dis_loss, "log", None, "upper right"),
        ("training_accuracy.png", f"Training Accuracy ({task_title})", "Training Accuracy (%)", en_tacc, dis_tacc, "linear", (40, 103), "lower right"),
        ("validation_accuracy.png", f"Validation Accuracy ({task_title})", "Validation Accuracy (%)", en_vacc, dis_vacc, "linear", (40, 103), "lower right"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=300)
        ax.plot(epochs, s_en, label=label_en, color="#1f77b4", lw=2.2)
        ax.plot(epochs, s_dis, label=label_dis, color="#d62728", ls="--", lw=2.2)
        ax.set_title(title, pad=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if yscale == "log":
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, which="both" if yscale == "log" else "major")
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc=lloc)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, fname), dpi=300)
        plt.close()

    # 2. OOD Generalization Line & Bar Charts
    en_ood_path = os.path.join(en_dir, "ood_evaluation.json")
    dis_ood_path = os.path.join(dis_dir, "ood_evaluation.json")
    if os.path.exists(en_ood_path) and os.path.exists(dis_ood_path):
        with open(en_ood_path) as f:
            en_ood_data = json.load(f)
        with open(dis_ood_path) as f:
            dis_ood_data = json.load(f)

        lengths = [int(L) for L in en_ood_data["lengths"]]
        en_accs = [float(en_ood_data["accuracies"][str(L)]) for L in lengths]
        dis_accs = [float(dis_ood_data["accuracies"][str(L)]) for L in lengths]
        train_max = tcfg["seq_len"]

        fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=300)
        ax.axvspan(lengths[0], train_max, color="#f1f5f9", alpha=1.0, zorder=0, label=f"Training Horizon ($L \\leq {train_max}$)")
        ax.axvline(x=train_max, color="#64748b", linestyle=":", linewidth=1.6, zorder=1)
        ax.plot(lengths, en_accs, marker="o", markersize=6.5, lw=2.2, color="#1f77b4", label=label_en, zorder=3)
        ax.plot(lengths, dis_accs, marker="^", markersize=6.5, lw=2.2, ls="--", color="#d62728", label=label_dis, zorder=3)
        ax.set_title(f"OOD Length Generalization ({task_title})", pad=12, fontweight="bold")
        ax.set_xlabel("Evaluation Sequence Length (Tokens)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(lengths)
        ax.set_ylim(15, 105)
        ax.grid(True, zorder=0)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "ood_length_generalization.png"), dpi=300)
        plt.close()

        bar_lengths = [L for L in [20, 40, 80, 120, 256] if L in lengths]
        bar_cats = [f"L={L}" for L in bar_lengths]
        b_en = [float(en_ood_data["accuracies"][str(L)]) for L in bar_lengths]
        b_dis = [float(dis_ood_data["accuracies"][str(L)]) for L in bar_lengths]

        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
        x = np.arange(len(bar_cats))
        w = 0.32
        r1 = ax.bar(x - w / 2, b_en, w, label=label_en, color="#1f77b4", edgecolor="#333333", alpha=0.9)
        r2 = ax.bar(x + w / 2, b_dis, w, label=label_dis, color="#d62728", edgecolor="#333333", alpha=0.9)
        for rects in (r1, r2):
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_title(f"{task_title}: Accuracy Across Sequence Horizons", pad=12, fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(bar_cats)
        ax.set_ylim(0, 118)
        ax.grid(axis="y", zorder=0)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "ood_summary_bar.png"), dpi=300)
        plt.close()

    # 3. Gate Dynamics (All 64 Neurons HAC Line Plots + Heatmaps + Attention Matrix)
    en_gate_path = os.path.join(en_dir, "gate_activations_len_40.json")
    dis_gate_path = os.path.join(dis_dir, "gate_activations_len_40.json")
    if os.path.exists(en_gate_path) and os.path.exists(dis_gate_path):
        with open(en_gate_path) as f:
            en_g = json.load(f)
        with open(dis_gate_path) as f:
            dis_g = json.load(f)

        T = len(en_g["iraw"])
        timesteps = np.arange(T)
        token_names = en_g.get("token_names", [str(x) for x in en_g["sequence"][:T]])[:T]
        boundary_steps = en_g.get("boundary_steps", [12, 28])

        line_specs = [
            ("iraw", "1_pre_activated_input_gate_iraw.png", "Pre-Activated Input Gate ($\\tilde{i}_t$)", "Pre-Activation $\\tilde{i}_t$"),
            ("igate", "2_post_activated_input_gate_igate.png", "Post-Activated Input Gate ($i_t = \\exp(\\tilde{i}_t)$)", "Gate Activation $i_t$"),
            ("fraw", "3_pre_activated_forget_gate_fraw.png", "Pre-Activated Forget Gate ($\\tilde{f}_t$)", "Pre-Activation $\\tilde{f}_t$"),
            ("fgate", "4_post_activated_forget_gate_fgate.png", "Post-Activated Forget Gate ($f_t$)", "Gate Activation $f_t$"),
            ("n_state", "5_normalizer_state_n.png", "Normalizer State Accumulation ($n_t$)", "Normalizer $n_t$"),
            ("c_state", "6_internal_cell_state_c.png", "Internal Cell State Dynamics ($c_t$)", "Cell State $c_t$"),
        ]

        for vkey, fname, vtitle, ylabel in line_specs:
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=300, sharex=True)
            for idx, (g_dict, sub_title) in enumerate([(en_g, label_en), (dis_g, label_dis)]):
                ax = axes[idx]
                mat = np.array(g_dict[vkey])  # [T, 64]
                clusters = get_cluster_assignments(np.array(g_dict["c_state"]), n_clusters=4)
                for n_idx in range(mat.shape[1]):
                    c_id = clusters[n_idx]
                    ax.plot(timesteps, mat[:, n_idx], color=CLUSTER_COLORS[c_id], alpha=0.35, lw=0.9)
                for c_id in range(4):
                    mask = clusters == c_id
                    if np.any(mask):
                        ax.plot(timesteps, np.mean(mat[:, mask], axis=1), color=CLUSTER_COLORS[c_id], lw=2.3, label=f"Cluster {c_id+1} (n={int(np.sum(mask))})")
                for b_step in boundary_steps:
                    if b_step < T:
                        ax.axvline(x=b_step, color="#dc2626", ls="--", lw=1.4)
                ax.set_title(sub_title, fontsize=11.5, fontweight="bold")
                ax.set_xlabel("Sequence Timestep $t$ (Token)")
                if idx == 0:
                    ax.set_ylabel(ylabel)
                tick_steps = list(range(0, T, 2))
                ax.set_xticks(tick_steps)
                ax.set_xticklabels([f"{t}\n({token_names[t]})" for t in tick_steps], fontsize=7.0)
                ax.grid(True)
                ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5, loc="upper left")
            fig.suptitle(f"{task_title} — {vtitle} Across All 64 Neurons", fontsize=13.5, fontweight="bold", y=1.01)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches="tight")
            plt.close()

        # 4-Gate Behaviour Heatmaps Grid
        gate_configs = [
            ("iraw", "Pre-Activated Input ($\\tilde{i}_t$)", "viridis", None, None),
            ("igate", "Post-Activated Input ($i_t$)", "YlGnBu", 0.0, None),
            ("fraw", "Pre-Activated Forget ($\\tilde{f}_t$)", "magma", None, None),
            ("fgate", "Post-Activated Forget ($f_t$)", "YlOrRd", 0.0, 1.0),
        ]
        tick_steps = list(range(0, T, 2))
        fig_grid, axes_grid = plt.subplots(4, 2, figsize=(12.5, 14.5), dpi=300, sharex=True, sharey=True)
        for row_idx, (vkey, vtitle, cmap, vmin, vmax) in enumerate(gate_configs):
            for col_idx, (g_dict, sub_title) in enumerate([(en_g, label_en), (dis_g, label_dis)]):
                ax = axes_grid[row_idx, col_idx]
                mat = np.array(g_dict[vkey])
                order = get_hac_order(np.array(g_dict["c_state"]))
                im = ax.imshow(mat[:, order].T, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
                for b_step in boundary_steps:
                    if b_step < T:
                        ax.axvline(x=b_step - 0.5, color="#dc2626", ls="--", lw=1.4)
                if row_idx == 0:
                    ax.set_title(sub_title, fontsize=11.5, fontweight="bold", pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f"{vtitle}\nNeuron (HAC)", fontsize=9.5, fontweight="bold")
                if row_idx == 3:
                    ax.set_xlabel("Timestep $t$ ($x_t$)")
                    ax.set_xticks(tick_steps)
                    ax.set_xticklabels([f"{t}\n({token_names[t]})" for t in tick_steps], fontsize=6.8)
                fig_grid.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig_grid.suptitle(f"{task_title} — Gate Behaviour Heatmaps Across All 64 Neurons", fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "gate_behaviour_heatmaps_4gates.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Attention Matrix Comparison
        fig_att, axes_att = plt.subplots(1, 2, figsize=(11.8, 5.4), dpi=300, sharey=True)
        for col_idx, (g_dict, sub_title) in enumerate([(en_g, label_en), (dis_g, label_dis)]):
            ax = axes_att[col_idx]
            alpha_mat = np.array(g_dict["alpha_matrix"])
            im = ax.imshow(alpha_mat, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0, origin="upper")
            for b_step in boundary_steps:
                if b_step < T:
                    ax.axvline(x=b_step - 0.5, color="#dc2626", ls="--", lw=1.2)
                    ax.axhline(y=b_step - 0.5, color="#dc2626", ls="--", lw=1.2)
            ax.set_title(sub_title, fontsize=11.5, fontweight="bold", pad=10)
            ax.set_xlabel("Historical Key Timestep $k$ ($x_k$)")
            ax.set_xticks(tick_steps)
            ax.set_xticklabels([f"{k}\n({token_names[k]})" for k in tick_steps], fontsize=6.8)
            if col_idx == 0:
                ax.set_ylabel("Query Timestep $t$ ($x_t$)")
                ax.set_yticks(tick_steps)
                ax.set_yticklabels([f"{t} ({token_names[t]})" for t in tick_steps], fontsize=7.0)
            fig_att.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention Weight $\\alpha(t, k)$")
        fig_att.suptitle(f"{task_title} — Empirical Pseudo-Softmax Attention Matrix $\\alpha(t, k)$", fontsize=13.5, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "attention_heatmap_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # 4. System Resources Comparison Chart
    en_res_path = os.path.join(en_dir, "system_resources.json")
    dis_res_path = os.path.join(dis_dir, "system_resources.json")
    if os.path.exists(en_res_path) and os.path.exists(dis_res_path):
        with open(en_res_path) as f:
            en_r = json.load(f)
        with open(dis_res_path) as f:
            dis_r = json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), dpi=300)
        labels = ["Forget Enabled", "Forget Disabled"]
        cols = ["#1f77b4", "#d62728"]

        # Subplot 1: Active Parameters
        p_vals = [en_r["parameters"]["active_params"], dis_r["parameters"]["active_params"]]
        b0 = axes[0].bar(labels, p_vals, color=cols, width=0.48, edgecolor="#333333")
        axes[0].set_title("Active Forward Parameters", fontweight="bold")
        axes[0].set_ylabel("Parameter Count")
        axes[0].set_ylim(0, max(p_vals) * 1.2)
        for rect in b0:
            h = rect.get_height()
            axes[0].annotate(f"{int(h):,}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

        # Subplot 2: Mean Epoch Time
        t_vals = [en_r["training_timing"]["mean_epoch_time_sec"], dis_r["training_timing"]["mean_epoch_time_sec"]]
        b1 = axes[1].bar(labels, t_vals, color=cols, width=0.48, edgecolor="#333333")
        axes[1].set_title("Mean Training Epoch Time", fontweight="bold")
        axes[1].set_ylabel("Seconds / Epoch")
        axes[1].set_ylim(0, max(t_vals) * 1.25)
        for rect in b1:
            h = rect.get_height()
            axes[1].annotate(f"{h:.3f}s", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

        # Subplot 3: Inference Latency (L=40 & L=256)
        inf_en_40 = en_r["inference_benchmark"]["L=40"]["mean_batch_latency_ms"]
        inf_dis_40 = dis_r["inference_benchmark"]["L=40"]["mean_batch_latency_ms"]
        inf_en_256 = en_r["inference_benchmark"]["L=256"]["mean_batch_latency_ms"]
        inf_dis_256 = dis_r["inference_benchmark"]["L=256"]["mean_batch_latency_ms"]
        x_inf = np.arange(2)
        w = 0.32
        r_en = axes[2].bar(x_inf - w / 2, [inf_en_40, inf_en_256], w, label="Forget Enabled", color="#1f77b4", edgecolor="#333333")
        r_dis = axes[2].bar(x_inf + w / 2, [inf_dis_40, inf_dis_256], w, label="Forget Disabled", color="#d62728", edgecolor="#333333")
        axes[2].set_title("Inference Batch Latency", fontweight="bold")
        axes[2].set_ylabel("Latency (ms)")
        axes[2].set_xticks(x_inf)
        axes[2].set_xticklabels(["L=40", "L=256"])
        axes[2].set_ylim(0, max(inf_en_256, inf_dis_256) * 1.25)
        axes[2].legend(frameon=True, fontsize=8.5)
        for rects in (r_en, r_dis):
            for rect in rects:
                h = rect.get_height()
                axes[2].annotate(f"{h:.1f}ms", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

        for ax in axes:
            ax.grid(axis="y", zorder=0)
        fig.suptitle(f"{task_title} — Parameter & Runtime Efficiency Comparison", fontsize=13.5, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "resource_efficiency_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Generated comparative visualizations for {task_title} in {fig_dir}")

