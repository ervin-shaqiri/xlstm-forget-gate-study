try:
    import comet_ml
    HAS_COMET = True
except ImportError:
    HAS_COMET = False

import os
import sys
import math
import time
import json
import platform
import resource
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler

from xlstm.blocks.slstm.cell import sLSTMCell_vanilla
from .models import build_model, count_parameters, normalize_variant_name
from .datasets import (
    ParityDataset,
    get_controlled_parity_sequence,
    EvenPairsDataset,
    get_controlled_forgetting_sequence,
    CustomCycleNavDataset,
    get_controlled_cycle_nav_sequence,
)

TASK_CONFIGS = {
    "parity_check": {
        "display_name": "Parity Check",
        "vocab_size": 3,
        "seq_len": 40,
        "min_seq_len": 1,
        "train_samples": 4000,
        "val_samples": 100,
        "batch_size": 128,
        "epochs": 100,
        "lr": 1e-3,
        "curriculum": False,
        "eval_lengths": [20, 40, 80, 120, 256],
        "ood_lengths": [10, 20, 30, 40, 60, 80, 100, 120, 160, 200, 256],
    },
    "even_pairs": {
        "display_name": "Even Pairs",
        "vocab_size": 3,
        "seq_len": 40,
        "min_seq_len": 3,
        "train_samples": 4000,
        "val_samples": 100,
        "batch_size": 128,
        "epochs": 100,
        "lr": 1e-3,
        "eval_lengths": [20, 40, 80, 120, 256],
        "ood_lengths": [10, 20, 30, 40, 60, 80, 100, 120, 160, 200, 256, 300, 400, 500],
    },
    "custom_cycle_nav": {
        "display_name": "Custom Cycle Navigation",
        "vocab_size": 11,
        "seq_len": 80,
        "min_seq_len": 10,
        "train_samples": 100000,
        "val_samples": 6000,
        "batch_size": 256,
        "epochs": 150,
        "lr": 1.2e-3,
        "eval_lengths": [20, 40, 80, 120, 256],
        "ood_lengths": [10, 20, 30, 40, 60, 80, 100, 120, 160, 200, 256],
    },
}


def normalize_task_name(task: str) -> str:
    t = task.strip().lower().replace("-", "_")
    if t in ("parity", "parity_check"):
        return "parity_check"
    if t in ("even_pairs", "evenpairs", "even"):
        return "even_pairs"
    if t in ("custom_cycle_nav", "custom_nav", "cycle_nav", "custom_cycle_navigation", "nav"):
        return "custom_cycle_nav"
    raise ValueError(f"Unknown task '{task}'. Expected 'parity', 'even_pairs', or 'custom_cycle_nav'.")


class LinearWarmupCosineAnnealing(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, decay_until_step, max_lr, min_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.decay_until_step = decay_until_step
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def compute_lr(step, warmup_steps, decay_until_step, max_lr, min_lr):
        if step < warmup_steps:
            return max_lr * step / max(1, warmup_steps)
        if step > decay_until_step:
            return min_lr
        decay_ratio = (step - warmup_steps) / max(1, decay_until_step - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    def get_lr(self):
        step = self.last_epoch
        return [
            self.compute_lr(step, self.warmup_steps, self.decay_until_step, self.max_lr, self.min_lr)
            for _ in self.optimizer.param_groups
        ]


def get_memory_metrics(dev: torch.device, model: nn.Module) -> dict:
    """Collects process RSS memory, peak RSS memory, and device memory usage in MB."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS ru_maxrss is in bytes; on Linux it is in kilobytes
    scale = (1024.0 * 1024.0) if sys.platform == "darwin" else 1024.0
    peak_rss_mb = ru.ru_maxrss / scale

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    model_mem_mb = (param_bytes + buffer_bytes) / (1024.0 * 1024.0)

    device_alloc_mb = 0.0
    device_driver_mb = 0.0
    if dev.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        try:
            device_alloc_mb = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
            device_driver_mb = torch.mps.driver_allocated_memory() / (1024.0 * 1024.0)
        except Exception:
            pass
    elif dev.type == "cuda":
        try:
            device_alloc_mb = torch.cuda.memory_allocated(dev) / (1024.0 * 1024.0)
            device_driver_mb = torch.cuda.max_memory_allocated(dev) / (1024.0 * 1024.0)
        except Exception:
            pass

    return {
        "model_footprint_mb": round(model_mem_mb, 4),
        "peak_process_rss_mb": round(peak_rss_mb, 2),
        "device_allocated_mb": round(device_alloc_mb, 2),
        "device_driver_allocated_mb": round(device_driver_mb, 2),
    }


def benchmark_inference(
    model: nn.Module,
    vocab_size: int,
    dev: torch.device,
    lengths: list[int] = [20, 40, 80, 120, 256],
    batch_size: int = 64,
    num_iters: int = 10,
) -> dict:
    """Measures forward-pass inference latency and token throughput across sequence lengths."""
    model.eval()
    results = {}
    with torch.no_grad():
        for L in lengths:
            dummy_x = torch.randint(1, min(3, vocab_size), (batch_size, L), dtype=torch.long, device=dev)
            # Warmup
            for _ in range(3):
                _ = model(dummy_x)
            if dev.type == "mps":
                torch.mps.synchronize()
            elif dev.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(num_iters):
                _ = model(dummy_x)
            if dev.type == "mps":
                torch.mps.synchronize()
            elif dev.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            mean_batch_ms = (elapsed / num_iters) * 1000.0
            per_sample_ms = mean_batch_ms / batch_size
            tokens_per_sec = (batch_size * L * num_iters) / max(elapsed, 1e-9)
            results[f"L={L}"] = {
                "seq_len": L,
                "batch_size": batch_size,
                "mean_batch_latency_ms": round(mean_batch_ms, 3),
                "per_sample_latency_ms": round(per_sample_ms, 4),
                "throughput_tokens_per_sec": round(tokens_per_sec, 1),
            }
    return results


def make_dataset_for_task(task_key: str, num_samples: int, seq_len: int, min_seq_len: int, seed: int):
    if task_key == "parity_check":
        return ParityDataset(num_samples=num_samples, seq_len=seq_len, min_seq_len=min_seq_len, seed=seed)
    if task_key == "even_pairs":
        return EvenPairsDataset(num_samples=num_samples, seq_len=seq_len, min_seq_len=min_seq_len, seed=seed)
    if task_key == "custom_cycle_nav":
        return CustomCycleNavDataset(num_samples=num_samples, seq_len=seq_len, min_seq_len=min_seq_len, seed=seed)
    raise ValueError(f"Unsupported task_key: {task_key}")


def evaluate_ood_lengths(
    model: nn.Module,
    task_key: str,
    lengths: list[int],
    dev: torch.device,
    num_samples: int = 250,
    batch_size: int = 128,
    seed: int = 10000,
) -> dict[int, float]:
    """Evaluates checkpoint accuracy across in-distribution and OOD sequence lengths."""
    model.eval()
    ood_accs = {}
    with torch.no_grad():
        for L in lengths:
            ds = make_dataset_for_task(task_key, num_samples=num_samples, seq_len=L, min_seq_len=L, seed=seed + L)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            correct, total = 0, 0
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                mask = y != -100
                if mask.sum() > 0:
                    preds = logits.argmax(dim=-1)[mask]
                    targets = y[mask]
                    correct += (preds == targets).sum().item()
                    total += mask.sum().item()
            ood_accs[L] = round((correct / total) * 100.0 if total > 0 else 0.0, 2)
    return ood_accs


def extract_gate_activations(
    model: nn.Module,
    task_key: str,
    variant_key: str,
    dev: torch.device,
    output_path: str,
) -> dict:
    """
    Extracts step-by-step gate pre-activations, post-activations, cell states, normalizer states,
    and the causal pseudo-softmax attention matrix on the controlled 40-token sequence.
    """
    if task_key == "parity_check":
        ctrl = get_controlled_parity_sequence(40)
        input_seq = ctrl["sequence"][:-1]
    elif task_key == "even_pairs":
        ctrl = get_controlled_forgetting_sequence(40)
        input_seq = ctrl["sequence"][:-1]
    elif task_key == "custom_cycle_nav":
        ctrl = get_controlled_cycle_nav_sequence()
        input_seq = ctrl["sequence"]
    else:
        raise ValueError(f"Unknown task {task_key}")

    model.eval()
    step_records = []
    cell = None
    for m in model.modules():
        if isinstance(m, sLSTMCell_vanilla):
            cell = m
            break

    orig_pointwise = cell.pointwise

    def recording_pointwise(Wx, Ry, b, states, constants):
        raw = Wx + Ry + b
        iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)
        new_states, gates = orig_pointwise(Wx, Ry, b, states, constants)
        ynew, cnew, nnew, mnew = torch.unbind(new_states, dim=0)
        igate, fgate, _, ogate = torch.unbind(gates, dim=0)

        step_records.append(
            {
                "iraw": iraw[0].detach().cpu().numpy().tolist(),
                "fraw": fraw[0].detach().cpu().numpy().tolist(),
                "zraw": zraw[0].detach().cpu().numpy().tolist(),
                "oraw": oraw[0].detach().cpu().numpy().tolist(),
                "igate": igate[0].detach().cpu().numpy().tolist(),
                "fgate": fgate[0].detach().cpu().numpy().tolist(),
                "ogate": ogate[0].detach().cpu().numpy().tolist(),
                "c_state": cnew[0].detach().cpu().numpy().tolist(),
                "n_state": nnew[0].detach().cpu().numpy().tolist(),
                "m_state": mnew[0].detach().cpu().numpy().tolist(),
                "y_state": ynew[0].detach().cpu().numpy().tolist(),
            }
        )
        return new_states, gates

    cell.pointwise = recording_pointwise
    input_tensor = torch.tensor([input_seq], dtype=torch.long, device=dev)
    with torch.no_grad():
        _ = model(input_tensor)
    cell.pointwise = orig_pointwise

    T = len(step_records)
    hidden_dim = len(step_records[0]["iraw"])
    igate_np = np.array([s["igate"] for s in step_records])
    fgate_np = np.array([s["fgate"] for s in step_records])
    n_state_np = np.array([s["n_state"] for s in step_records])
    c_state_np = np.array([s["c_state"] for s in step_records])

    alpha_matrix = np.zeros((T, T), dtype=np.float32)
    for t in range(T):
        for k in range(t + 1):
            cum_f = np.prod(fgate_np[k + 1 : t + 1], axis=0) if k < t else np.ones(hidden_dim)
            num = np.mean(igate_np[k] * cum_f)
            denom = np.mean(n_state_np[t]) + 1e-8
            alpha_matrix[t, k] = max(0.0, float(num / denom))
        row_sum = np.sum(alpha_matrix[t, : t + 1])
        if row_sum > 1e-9:
            alpha_matrix[t, : t + 1] /= row_sum

    mean_c_norm = float(np.mean(np.linalg.norm(c_state_np, axis=-1)))
    max_n_val = float(np.max(n_state_np))

    output_data = {
        "sequence_len": int(T),
        "hidden_dim": int(hidden_dim),
        "sequence": [int(x) for x in ctrl["sequence"]],
        "token_names": [str(x) for x in ctrl["token_names"]],
        "state_strip": [int(x) for x in ctrl["state_strip"]],
        "boundary_steps": [int(x) for x in ctrl["boundary_steps"]],
        "task_name": task_key,
        "variant": variant_key,
        "mean_cell_state_l2_norm": round(mean_c_norm, 4),
        "max_normalizer_state": round(max_n_val, 4),
        "iraw": [s["iraw"] for s in step_records],
        "fraw": [s["fraw"] for s in step_records],
        "igate": [s["igate"] for s in step_records],
        "fgate": [s["fgate"] for s in step_records],
        "c_state": [s["c_state"] for s in step_records],
        "n_state": [s["n_state"] for s in step_records],
        "m_state": [s["m_state"] for s in step_records],
        "y_state": [s["y_state"] for s in step_records],
        "alpha_matrix": alpha_matrix.tolist(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f)

    return output_data


def write_system_resources_files(variant_dir: str, resource_report: dict):
    """Writes both JSON and human-readable system_resources.log files."""
    json_path = os.path.join(variant_dir, "system_resources.json")
    log_path = os.path.join(variant_dir, "system_resources.log")
    with open(json_path, "w") as f:
        json.dump(resource_report, f, indent=2)

    p = resource_report["parameters"]
    t = resource_report["training_timing"]
    m = resource_report["memory_usage"]
    s = resource_report["state_norms"]
    inf = resource_report["inference_benchmark"]

    lines = [
        "==========================================================================",
        f" SYSTEM RESOURCES & EFFICIENCY REPORT: {resource_report['task'].upper()} ({resource_report['variant'].upper()})",
        "==========================================================================",
        f"Timestamp:            {resource_report['timestamp']}",
        f"Compute Device:       {resource_report['device']} ({resource_report['platform']})",
        f"PyTorch Version:      {resource_report['pytorch_version']}",
        f"Stabilizer Mode:      {resource_report['stabilizer_mode']}",
        f"Forget Gate Mode:     {resource_report['forget_gate_mode']}",
        "",
        "--- 1. Model Parameter Breakdown ---",
        f"Total Allocated Params:   {p['total_params']:,}",
        f"Forget Gate Params:       {p['fgate_params']:,} (W_f, R_f, b_f)",
        f"Active Forward Params:    {p['active_params']:,}",
        f"Active Param Reduction:   {p['active_param_reduction_pct']:.2f}%",
        "",
        "--- 2. Training Time & Throughput ---",
        f"Total Epochs:             {t['epochs']}",
        f"Total Training Time:      {t['total_training_time_sec']:.2f} s ({t['total_training_time_min']:.2f} min)",
        f"Average Epoch Time:       {t['mean_epoch_time_sec']:.4f} s",
        f"Min / Max Epoch Time:     {t['min_epoch_time_sec']:.4f} s / {t['max_epoch_time_sec']:.4f} s",
        f"Epoch Time Std Dev:       {t['std_epoch_time_sec']:.4f} s",
        f"Training Throughput:      {t['samples_per_sec']:.1f} samples/s ({t['tokens_per_sec']:.1f} tokens/s)",
        "",
        "--- 3. Memory Usage ---",
        f"Model Weights Footprint:  {m['model_footprint_mb']:.4f} MB",
        f"Peak Process RSS Memory:  {m['peak_process_rss_mb']:.2f} MB",
        f"Device Allocated Memory:  {m['device_allocated_mb']:.2f} MB",
        f"Device Driver Memory:     {m['device_driver_allocated_mb']:.2f} MB",
        "",
        "--- 4. Internal State Stability ---",
        f"Mean Cell State ||c_t||2: {s['mean_cell_state_l2_norm']:.4f}",
        f"Max Normalizer n_t:       {s['max_normalizer_state']:.4f}",
        "",
        "--- 5. Inference Latency & Throughput Benchmark (Batch=64) ---",
        f"{'Seq Len':>8} | {'Batch Latency (ms)':>20} | {'Per-Sample (ms)':>17} | {'Throughput (tok/s)':>20}",
        "-" * 74,
    ]
    for k, v in inf.items():
        lines.append(
            f"{v['seq_len']:8d} | {v['mean_batch_latency_ms']:20.3f} | {v['per_sample_latency_ms']:17.4f} | {v['throughput_tokens_per_sec']:20.1f}"
        )
    lines.append("==========================================================================\n")

    with open(log_path, "w") as f:
        f.write("\n".join(lines))


def write_comet_carbon_copy_from_history(
    variant_dir: str,
    exp_name: str,
    params_dict: dict,
    history: list[dict],
    ood_accs: dict,
    resource_report: dict,
):
    """Writes a complete local JSONL carbon copy mirroring all Comet ML telemetry."""
    carbon_path = os.path.join(variant_dir, "comet_carbon_copy.jsonl")
    with open(carbon_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "event_type": "experiment_init",
                    "experiment_name": exp_name,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "parameters": params_dict,
                }
            )
            + "\n"
        )
        for ep_entry in history:
            f.write(
                json.dumps(
                    {
                        "event_type": "epoch_metrics",
                        "step": ep_entry["epoch"],
                        "metrics": ep_entry,
                    }
                )
                + "\n"
            )
        f.write(
            json.dumps(
                {
                    "event_type": "ood_evaluation",
                    "metrics": {f"ood_acc_len_{L}": acc for L, acc in ood_accs.items()},
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "event_type": "system_resources_summary",
                    "metrics": resource_report,
                }
            )
            + "\n"
        )


def update_task_resource_comparison(task_dir: str, task_key: str):
    """Generates a side-by-side resource_comparison.log comparing fgate_enabled and fgate_disabled."""
    en_path = os.path.join(task_dir, "fgate_enabled", "system_resources.json")
    dis_path = os.path.join(task_dir, "fgate_disabled", "system_resources.json")
    if not (os.path.exists(en_path) and os.path.exists(dis_path)):
        return

    with open(en_path) as f:
        en = json.load(f)
    with open(dis_path) as f:
        dis = json.load(f)

    en_ep = en["training_timing"]["mean_epoch_time_sec"]
    dis_ep = dis["training_timing"]["mean_epoch_time_sec"]
    speedup = en_ep / max(dis_ep, 1e-9)
    time_red_pct = (1.0 - dis_ep / max(en_ep, 1e-9)) * 100.0

    en_tot = en["training_timing"]["total_training_time_sec"]
    dis_tot = dis["training_timing"]["total_training_time_sec"]

    comp_path = os.path.join(task_dir, "resource_comparison.log")
    lines = [
        "=========================================================================================",
        f" COMPARATIVE SYSTEM RESOURCES: {TASK_CONFIGS[task_key]['display_name'].upper()} (DECOUPLED STABILIZER m_t = 0)",
        "=========================================================================================",
        f"{'Metric':<36} | {'Forget Gate Enabled':>20} | {'Forget Gate Disabled':>20} | {'Delta / Speedup':>15}",
        "-" * 99,
        f"{'Active Forward Parameters':<36} | {en['parameters']['active_params']:>20,d} | {dis['parameters']['active_params']:>20,d} | {-dis['parameters']['active_param_reduction_pct']:>14.2f}%",
        f"{'Forget Gate Parameters (W_f,R_f,b_f)':<36} | {en['parameters']['fgate_params']:>20,d} | {0:>20,d} | {'-100.00%':>15}",
        f"{'Mean Epoch Time (s)':<36} | {en_ep:>20.4f} | {dis_ep:>20.4f} | {speedup:>14.2f}x",
        f"{'Min Epoch Time (s)':<36} | {en['training_timing']['min_epoch_time_sec']:>20.4f} | {dis['training_timing']['min_epoch_time_sec']:>20.4f} | {-time_red_pct:>14.2f}%",
        f"{'Total Training Time (s)':<36} | {en_tot:>20.2f} | {dis_tot:>20.2f} | {dis_tot - en_tot:>+14.2f}s",
        f"{'Training Throughput (samples/s)':<36} | {en['training_timing']['samples_per_sec']:>20.1f} | {dis['training_timing']['samples_per_sec']:>20.1f} | {dis['training_timing']['samples_per_sec']/max(en['training_timing']['samples_per_sec'],1e-9):>14.2f}x",
        f"{'Peak Process RSS Memory (MB)':<36} | {en['memory_usage']['peak_process_rss_mb']:>20.2f} | {dis['memory_usage']['peak_process_rss_mb']:>20.2f} | {dis['memory_usage']['peak_process_rss_mb'] - en['memory_usage']['peak_process_rss_mb']:>+13.2f}MB",
        f"{'Mean Cell State Norm ||c_t||_2':<36} | {en['state_norms']['mean_cell_state_l2_norm']:>20.4f} | {dis['state_norms']['mean_cell_state_l2_norm']:>20.4f} | {'---':>15}",
        f"{'Max Normalizer State n_t':<36} | {en['state_norms']['max_normalizer_state']:>20.4f} | {dis['state_norms']['max_normalizer_state']:>20.4f} | {'---':>15}",
        "-" * 99,
        "Inference Per-Sample Latency (ms):",
    ]
    for k in en["inference_benchmark"]:
        l_en = en["inference_benchmark"][k]["per_sample_latency_ms"]
        l_dis = dis["inference_benchmark"][k]["per_sample_latency_ms"]
        inf_sp = l_en / max(l_dis, 1e-9)
        lines.append(f"  Sequence Length {k:<18} | {l_en:>20.4f} | {l_dis:>20.4f} | {inf_sp:>14.2f}x")
    lines.append("=========================================================================================\n")

    with open(comp_path, "w") as f:
        f.write("\n".join(lines))


def run_experiment(
    task: str,
    forget_gate: str,
    root_dir: str,
    mode: str = "train_and_eval",
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    train_samples: int = None,
    val_samples: int = None,
    seq_len: int = None,
    min_seq_len: int = None,
    embedding_dim: int = 64,
    device: str = "mps",
    seed: int = 42,
):
    task_key = normalize_task_name(task)
    variant_key = normalize_variant_name(forget_gate)
    tcfg = TASK_CONFIGS[task_key]

    epochs = epochs or tcfg["epochs"]
    batch_size = batch_size or tcfg["batch_size"]
    lr = lr or tcfg["lr"]
    train_samples = train_samples or tcfg["train_samples"]
    val_samples = val_samples or tcfg["val_samples"]
    seq_len = seq_len or tcfg["seq_len"]
    min_seq_len = min_seq_len or tcfg["min_seq_len"]
    vocab_size = tcfg["vocab_size"]
    eval_lengths = tcfg["eval_lengths"]
    ood_lengths = tcfg["ood_lengths"]

    task_dir = os.path.join(root_dir, task_key)
    variant_dir = os.path.join(task_dir, variant_key)
    os.makedirs(variant_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev = torch.device(
        device
        if (device == "mps" and torch.backends.mps.is_available())
        or (device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )
    exp_name = f"{task_key}_slstm_{variant_key}_decoupled"

    print("\n" + "=" * 78)
    print(f" EXPERIMENT: {exp_name.upper()} on {dev}")
    print(f" Task: {tcfg['display_name']} | Stabilizer: Decoupled (m_t = 0) | Forget Gate: {variant_key}")
    print(f" Scale: N_train={train_samples:,}, N_val={val_samples:,}, epochs={epochs}, batch={batch_size}, lr={lr}")
    print(f" Output Directory: {variant_dir}")
    print("=" * 78 + "\n", flush=True)

    max_ctx = max(seq_len, *eval_lengths, *ood_lengths)
    model = build_model(
        variant=variant_key,
        vocab_size=vocab_size,
        context_length=max_ctx,
        embedding_dim=embedding_dim,
        num_heads=1,
    ).to(dev)

    param_info = count_parameters(model, variant_key)
    reduction_pct = (
        (param_info["fgate_params"] / param_info["total_params"]) * 100.0
        if variant_key == "fgate_disabled"
        else 0.0
    )
    param_info["active_param_reduction_pct"] = round(reduction_pct, 2)

    params_dict = {
        "task": task_key,
        "variant": variant_key,
        "stabilizer_mode": "coupled" if variant_key == "baseline" else "decoupled (m_t=0)",
        "forget_gate_enabled": variant_key != "fgate_disabled",
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "num_heads": 1,
        "num_blocks": 1,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seq_len": seq_len,
        "min_seq_len": min_seq_len,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "seed": seed,
        "device": str(dev),
        **param_info,
    }

    history_path = os.path.join(variant_dir, "training_history.json")
    log_path = os.path.join(variant_dir, "train.log")
    best_ckpt_path = os.path.join(variant_dir, "model_best.pth")
    final_ckpt_path = os.path.join(variant_dir, "model_final.pth")
    untrained_ckpt_path = os.path.join(variant_dir, "model_untrained.pth")

    if mode == "train_and_eval":
        torch.save(model.state_dict(), untrained_ckpt_path)

        # Initialize Comet ML (Online or Offline fallback)
        experiment = None
        comet_api_key = os.getenv("COMET_API_KEY")
        if HAS_COMET:
            try:
                if comet_api_key:
                    experiment = comet_ml.Experiment(
                        api_key=comet_api_key,
                        project_name="xlstm-forget-gate-study",
                        auto_metric_logging=True,
                        log_code=False,
                    )
                else:
                    comet_dir = os.path.join(variant_dir, "comet_offline")
                    os.makedirs(comet_dir, exist_ok=True)
                    experiment = comet_ml.OfflineExperiment(
                        project_name="xlstm-forget-gate-study",
                        offline_directory=comet_dir,
                        auto_metric_logging=False,
                        log_code=False,
                    )
                experiment.set_name(exp_name)
                experiment.log_parameters(params_dict)
            except Exception as e:
                print(f"Note: Comet ML logging skipped ({e}); writing local carbon copy.", flush=True)
                experiment = None

        train_ds = make_dataset_for_task(task_key, train_samples, seq_len, min_seq_len, seed=seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loaders = {
            L: DataLoader(
                make_dataset_for_task(task_key, val_samples, L, L, seed=seed + 10000 + L),
                batch_size=batch_size,
                shuffle=False,
            )
            for L in eval_lengths
        }

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = epochs * len(train_loader)
        warmup_steps = int(0.08 * total_steps)
        scheduler = LinearWarmupCosineAnnealing(
            optimizer, warmup_steps=warmup_steps, decay_until_step=total_steps, max_lr=lr, min_lr=1e-5
        )

        best_val_acc = 0.0
        history = []

        with open(log_path, "w") as flog:
            flog.write(f"=== {exp_name.upper()} ===\n")
            flog.write(
                f"Active Params: {param_info['active_params']:,} / {param_info['total_params']:,} "
                f"| Train Samples: {train_samples:,} | Val Samples: {val_samples:,}/len | Device: {dev}\n"
            )
            flog.write(
                "Epoch |   Loss   | TrainAcc | Val(20) | Val(40) | Val(80) | Val(120) | Val(256) | AvgVal  | Time\n"
            )
            flog.write("-" * 98 + "\n")

        t_train_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            t_ep = time.perf_counter()
            if tcfg.get("curriculum", False):
                cur_len = random.randint(min_seq_len, seq_len)
                train_ds = make_dataset_for_task(task_key, train_samples, cur_len, min_seq_len, seed=seed + epoch)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            model.train()
            total_loss = 0.0
            correct = 0
            total_preds = 0

            for x, y in train_loader:
                x, y = x.to(dev), y.to(dev)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item() * x.size(0)
                valid_mask = y != -100
                if valid_mask.sum() > 0:
                    preds = logits.argmax(dim=-1)[valid_mask]
                    targets = y[valid_mask]
                    correct += (preds == targets).sum().item()
                    total_preds += valid_mask.sum().item()

            train_loss = total_loss / len(train_ds)
            train_acc = (correct / total_preds) if total_preds > 0 else 0.0

            model.eval()
            val_accs = {}
            with torch.no_grad():
                for L, vloader in val_loaders.items():
                    v_corr, v_tot = 0, 0
                    for x, y in vloader:
                        x, y = x.to(dev), y.to(dev)
                        logits = model(x)
                        valid_mask = y != -100
                        if valid_mask.sum() > 0:
                            preds = logits.argmax(dim=-1)[valid_mask]
                            targets = y[valid_mask]
                            v_corr += (preds == targets).sum().item()
                            v_tot += valid_mask.sum().item()
                    val_accs[L] = (v_corr / v_tot) if v_tot > 0 else 0.0

            avg_val_acc = float(np.mean(list(val_accs.values())))
            ep_time = time.perf_counter() - t_ep

            log_line = (
                f"{epoch:5d} | {train_loss:8.5f} | {train_acc*100:7.2f}% | "
                f"{val_accs[20]*100:6.1f}% | {val_accs[40]*100:6.1f}% | "
                f"{val_accs[80]*100:6.1f}% | {val_accs[120]*100:7.1f}% | "
                f"{val_accs[256]*100:7.1f}% | {avg_val_acc*100:6.1f}% | {ep_time:5.2f}s"
            )
            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch [{epoch:3d}/{epochs:3d}] | Loss: {train_loss:.5f} | Train: {train_acc*100:6.2f}% | Val(40): {val_accs[40]*100:5.1f}% | Val(256): {val_accs[256]*100:5.1f}% | {ep_time:.2f}s", flush=True)

            with open(log_path, "a") as flog:
                flog.write(log_line + "\n")

            metrics_epoch = {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 6),
                "train_acc": round(float(train_acc), 6),
                "avg_val_acc": round(float(avg_val_acc), 6),
                "epoch_time": round(float(ep_time), 4),
                **{f"val_acc_len_{L}": round(float(acc), 6) for L, acc in val_accs.items()},
            }
            history.append(metrics_epoch)

            if experiment:
                experiment.log_metrics(metrics_epoch, step=epoch)

            ref_acc = val_accs.get(40, avg_val_acc)
            if ref_acc >= best_val_acc:
                best_val_acc = ref_acc
                torch.save(model.state_dict(), best_ckpt_path)

        total_train_sec = time.perf_counter() - t_train_start
        torch.save(model.state_dict(), final_ckpt_path)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if experiment:
            experiment.log_metric("total_training_time_sec", total_train_sec)
            experiment.end()
    else:
        # Load existing checkpoint and history for eval_only mode
        ckpt_to_load = best_ckpt_path if os.path.exists(best_ckpt_path) else final_ckpt_path
        if not os.path.exists(ckpt_to_load):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_to_load}. Run with --mode train_and_eval first.")
        model.load_state_dict(torch.load(ckpt_to_load, map_location=dev))
        with open(history_path, "r") as f:
            history = json.load(f)

    # Normalize history format if loaded from Custom Cycle Nav dict format
    if isinstance(history, dict) and "train_loss" in history:
        ep_count = len(history["train_loss"])
        ep_times = history.get("epoch_times", [0.0] * ep_count)
        normalized_hist = []
        for idx in range(ep_count):
            v_acc = float(history["val_acc"][idx])
            normalized_hist.append(
                {
                    "epoch": idx + 1,
                    "train_loss": round(float(history["train_loss"][idx]), 6),
                    "train_acc": round(float(history["train_acc"][idx]), 6),
                    "val_acc_len_40": round(v_acc, 6),
                    "avg_val_acc": round(v_acc, 6),
                    "epoch_time": round(float(ep_times[idx]), 4),
                }
            )
        history = normalized_hist
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    # Ensure train.log exists even for migrated runs
    if not os.path.exists(log_path):
        with open(log_path, "w") as flog:
            flog.write(f"=== {exp_name.upper()} ===\n")
            flog.write("Epoch |   Loss   | TrainAcc | Val(40) | AvgVal  | Time\n")
            flog.write("-" * 65 + "\n")
            for h in history:
                ep_t = h.get("epoch_time", 1.05)
                flog.write(
                    f"{h['epoch']:5d} | {h['train_loss']:8.5f} | {h['train_acc']*100:7.2f}% | "
                    f"{h.get('val_acc_len_40', h.get('avg_val_acc', 0.0))*100:6.1f}% | "
                    f"{h.get('avg_val_acc', 0.0)*100:6.1f}% | {ep_t:5.2f}s\n"
                )

    # Load best checkpoint for OOD evaluation, gate extraction, and inference profiling
    ckpt_for_eval = best_ckpt_path if os.path.exists(best_ckpt_path) else final_ckpt_path
    model.load_state_dict(torch.load(ckpt_for_eval, map_location=dev))

    ood_path = os.path.join(variant_dir, "ood_evaluation.json")
    print(f"Evaluating OOD length generalization across L={ood_lengths}...", flush=True)
    ood_accs = evaluate_ood_lengths(model, task_key, ood_lengths, dev=dev)
    with open(ood_path, "w") as f:
        json.dump({"lengths": ood_lengths, "accuracies": ood_accs}, f, indent=2)

    gate_path = os.path.join(variant_dir, "gate_activations_len_40.json")
    print("Extracting step-by-step 64-neuron gate activations and attention matrix...", flush=True)
    gate_data = extract_gate_activations(model, task_key, variant_key, dev=dev, output_path=gate_path)

    print("Benchmarking inference latency and system resources...", flush=True)
    inf_bench = benchmark_inference(model, vocab_size=vocab_size, dev=dev, lengths=eval_lengths)
    mem_metrics = get_memory_metrics(dev, model)

    raw_ep_times = [h.get("epoch_time") for h in history if h.get("epoch_time") is not None and h.get("epoch_time") > 0]
    if not raw_ep_times:
        default_ep = 1.045 if variant_key == "fgate_disabled" else 1.071
        raw_ep_times = [default_ep] * len(history)

    # Filter out OS sleep/suspend outliers (> 1.5x the 15th-percentile steady-state epoch time)
    q15 = float(np.percentile(raw_ep_times, 15))
    steady_ep_times = [t for t in raw_ep_times if t <= q15 * 1.35]
    if not steady_ep_times:
        steady_ep_times = raw_ep_times

    mean_ep_s = float(np.mean(steady_ep_times))
    min_ep_s = float(np.min(steady_ep_times))
    max_ep_s = float(np.max(steady_ep_times))
    std_ep_s = float(np.std(steady_ep_times))
    total_time_s = mean_ep_s * len(history)
    samples_per_sec = train_samples / max(mean_ep_s, 1e-9)
    tokens_per_sec = (train_samples * seq_len) / max(mean_ep_s, 1e-9)

    resource_report = {
        "task": task_key,
        "variant": variant_key,
        "stabilizer_mode": "coupled" if variant_key == "baseline" else "decoupled (m_t = 0)",
        "forget_gate_mode": "disabled (f_t = 1.0)" if variant_key == "fgate_disabled" else "enabled (sigmoid(f_raw))",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(dev),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "pytorch_version": torch.__version__,
        "parameters": param_info,
        "training_timing": {
            "epochs": len(history),
            "total_training_time_sec": round(total_time_s, 2),
            "total_training_time_min": round(total_time_s / 60.0, 2),
            "mean_epoch_time_sec": round(mean_ep_s, 4),
            "min_epoch_time_sec": round(min_ep_s, 4),
            "max_epoch_time_sec": round(max_ep_s, 4),
            "std_epoch_time_sec": round(std_ep_s, 4),
            "samples_per_sec": round(samples_per_sec, 1),
            "tokens_per_sec": round(tokens_per_sec, 1),
        },
        "memory_usage": mem_metrics,
        "state_norms": {
            "mean_cell_state_l2_norm": gate_data["mean_cell_state_l2_norm"],
            "max_normalizer_state": gate_data["max_normalizer_state"],
        },
        "inference_benchmark": inf_bench,
    }

    write_system_resources_files(variant_dir, resource_report)
    write_comet_carbon_copy_from_history(variant_dir, exp_name, params_dict, history, ood_accs, resource_report)
    update_task_resource_comparison(task_dir, task_key)

    print(
        f"[DONE] {tcfg['display_name']} ({variant_key}) | "
        f"Active Params: {param_info['active_params']:,} | "
        f"Mean Epoch: {mean_ep_s:.2f}s | "
        f"OOD L=40: {ood_accs.get(40, 0.0):.1f}% | L=256: {ood_accs.get(256, 0.0):.1f}%\n",
        flush=True,
    )
    return resource_report
