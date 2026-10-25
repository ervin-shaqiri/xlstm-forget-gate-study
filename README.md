# Decoupled Stabilizer sLSTM: Forget Gate Ablation Study on Formal State-Tracking Tasks

This repository contains a controlled empirical and mechanistic study of the **Scalar LSTM (sLSTM)** from the Extended Long Short-Term Memory (**xLSTM**) architecture (Beck et al., 2024) with a **Decoupled Stabilizer ($m_t = 0$)** across three formal state-tracking benchmarks from the Chomsky hierarchy (Deletang et al., 2023):

1. **Parity Check (`parity_check/`)** — Non-star-free regular language ($\mathbb{Z}_2$ modular counting).
2. **Even Pairs (`even_pairs/`)** — Star-free local regular language ($AC^0$ boundary comparison).
3. **Custom Cycle Navigation (`custom_cycle_nav/`)** — Non-commutative finite-state permutation group transitions with explicit state-reset commands ($S_5$ transitions over $k=5$ states).

All experiments compare two surgically isolated recurrent cell variants under identical training conditions, random seeds (`seed=42`), and model dimensions ($d_{\text{model}} = 64$, 1 sLSTM block, 4 heads, head dimension $d_h = 16$):
- **Forget Gate Enabled (`fgate_enabled`)**: Decoupled stabilizer $m_t = 0$ with a learned sigmoid forget gate $f_t = \sigma(\tilde{f}_t)$ and exponential input gate $i_t = \exp(\tilde{i}_t)$.
- **Forget Gate Disabled (`fgate_disabled`)**: Decoupled stabilizer $m_t = 0$ with the forget gate identically fixed to unity ($f_t \equiv 1.0$) and its projection parameters ($W_f, R_f, b_f$) completely removed from the active graph.

---

## 1. Theoretical Formulation & Purpose of the Study

### 1.1 Standard Coupling Trap in Official sLSTM
In the standard official sLSTM cell (`xlstm`), raw exponential gates are numerically stabilized via a running log-domain stabilizer state $m_t$:

$$m_t = \max\left(\log f_t + m_{t-1},\; \tilde{i}_t\right), \qquad i'_t = \exp(\tilde{i}_t - m_t), \qquad f'_t = \exp(\log f_t + m_{t-1} - m_t)$$

When ablating the forget gate by setting $f_t \equiv 1$ ($\log f_t = 0$) while retaining the standard stabilizer recurrence $m_t = \max(m_{t-1}, \tilde{i}_t)$, the stabilizer $m_t$ becomes a monotonically non-decreasing running maximum. This couples the effective forget gate $f'_t = \exp(m_{t-1} - m_t)$ to every new input spike $\tilde{i}_t > m_{t-1}$, creating an unintended synthetic forgetting mechanism even when $f_t \equiv 1$.

### 1.2 Decoupled Stabilizer ($m_t = 0$) & Pseudo-Softmax Memory Dilution
To isolate the true computational role of the forget gate $f_t$ without confounding stabilizer coupling, both variants in this repository decouple the stabilizer by fixing $m_t = 0$ (clamping pre-activation input gates $\tilde{i}_t \le 10.0$ to prevent `float32` overflow):

$$\begin{aligned}
i_t &= \exp(\tilde{i}_t) \\
f_t &= \begin{cases} \sigma(\tilde{f}_t) & \text{(Forget Gate Enabled: \texttt{fgate\_enabled})} \\ 1.0 & \text{(Forget Gate Disabled: \texttt{fgate\_disabled})} \end{cases} \\
c_t &= f_t \odot c_{t-1} + i_t \odot z_t \\
n_t &= f_t \odot n_{t-1} + i_t \\
h_t &= o_t \odot \frac{c_t}{\max(n_t, \epsilon)}
\end{aligned}$$

Unrolling the normalized readout $c_t / n_t$ when the forget gate is disabled ($f_t \equiv 1.0$) reveals an exact **monotonically accumulating exponential weighted average**:

$$\frac{c_T}{n_T} = \frac{\sum_{t=1}^{T} \exp(\tilde{i}_t) z_t}{n_0 + \sum_{t=1}^{T} \exp(\tilde{i}_t)} = \sum_{t=1}^{T} \alpha_{T,t} z_t, \qquad \text{where } \alpha_{T,t} = \frac{\exp(\tilde{i}_t)}{n_T}$$

This formulation exposes two core phenomena investigated in this repository:
1. **Pseudo-Softmax Overwriting vs. Multiplicative Reset**: With $f_t \equiv 1.0$, an sLSTM can still overwrite past states *temporarily* by firing exponentially larger input gates $\tilde{i}_t > \log n_{t-1}$ (driving $\alpha_{t,t} \to 1$ and diluting prior history by $n_{t-1}/n_t \to 0$). This allows the Forget Gate Disabled model to reach **99.6%–100.0% in-distribution accuracy** ($L \le 40$) across all three tasks, including Parity Check and Custom Cycle Navigation.
2. **Irreversible Normalizer Explosion ($n_t \to \infty$)**: Because $i_t = \exp(\tilde{i}_t) > 0$ and $f_t \equiv 1.0$, the normalizer $n_T = \sum_{t=1}^T \exp(\tilde{i}_t)$ grows strictly monotonically with sequence length $T$. On tasks requiring unbounded modular state flips (**Parity Check**) or repeated hard state resets (**Custom Cycle Navigation**), $n_t$ saturates the dynamic range as $T$ extends out-of-distribution ($L \ge 80$), causing length generalization to collapse to chance while the Forget Gate Enabled model maintains **100.0% accuracy up to $L=256$**. Conversely, on **Even Pairs**—which only requires comparing the first and last tokens—the Forget Gate Disabled model achieves **100.0% accuracy across all lengths up to $L=500$** while eliminating **14.10% of active parameters** and training **2.4% faster**.

---

## 2. Repository Structure

```text
xlstm-forget-gate-study/
├── main.py                              # Unified CLI entry point for training, evaluation, and plotting
├── README.md                            # Study overview, CLI documentation, and consolidated results table
├── pyproject.toml                       # Python dependencies and project metadata
├── uv.lock                              # Reproducible dependency lockfile
├── src/                                 # Core modularity and experiment implementation
│   ├── __init__.py
│   ├── xlstm/                           # Official xLSTM library (slstm_block, slstm_cell, mLSTM, etc.)
│   ├── models/
│   │   ├── cell_variants.py             # CustomDecoupledSLSTMCell (m_t=0, fgate_enabled vs fgate_disabled)
│   │   └── model_builder.py             # xLSTM sequence classifier builder and active parameter accounting
│   ├── datasets/
│   │   ├── parity_dataset.py            # Parity Check dataset & exact DFA step-by-step state tracker
│   │   ├── even_pairs_dataset.py        # Even Pairs dataset & boundary state tracker
│   │   └── custom_cycle_nav_dataset.py  # 5-State Custom Cycle Navigation dataset (+1, -1, +2, NOP, RESET)
│   ├── trainer.py                       # Training loop, Comet ML + local carbon-copy logger, resource profiler
│   └── visualization.py                 # 64-neuron gate heatmaps, state norms, training curves, resource plots
├── parity_check/                        # Task 1: Parity Check (Z_2 modular counter)
│   ├── fgate_enabled/                   # Decoupled sLSTM (m_t = 0, f_t = sigmoid) checkpoints & logs
│   ├── fgate_disabled/                  # Decoupled sLSTM (m_t = 0, f_t = 1.0) checkpoints & logs
│   ├── figures/                         # Comparative training, OOD, 64-neuron gate & resource plots
│   └── resource_comparison.log          # Side-by-side parameter, training time, memory & latency comparison
├── even_pairs/                          # Task 2: Even Pairs (first-vs-last boundary comparison)
│   ├── fgate_enabled/
│   ├── fgate_disabled/
│   ├── figures/
│   └── resource_comparison.log
└── custom_cycle_nav/                    # Task 3: Custom Cycle Navigation (5-state cyclic DFA with Reset=4)
    ├── fgate_enabled/
    ├── fgate_disabled/
    ├── figures/
    └── resource_comparison.log
```

### Artifact Files Inside Each `<task>/<variant>/` Directory
Each variant folder (`fgate_enabled/` and `fgate_disabled/`) contains:
- `model_untrained.pth`: Initial model weights prior to optimization (`seed=42`).
- `model_best.pth`: Checkpoint achieving the highest validation accuracy during training.
- `model_final.pth`: Final checkpoint at the end of training.
- `train.log`: Human-readable epoch-by-epoch training, validation, and OOD generalization log.
- `comet_carbon_copy.jsonl`: Complete local JSON-Lines carbon copy of every event sent to Comet ML (`hyperparameters`, per-epoch `epoch_metrics`, `ood_evaluation`, and `system_resources`).
- `training_history.json`: Full per-epoch history of training loss/accuracy, multi-length validation metrics ($L \in \{20, 40, 80, 120\}$), and epoch durations.
- `ood_evaluation.json`: Out-of-distribution generalization accuracy evaluated across sequence lengths up to $L=256$ (and $L=500$ for Even Pairs).
- `system_resources.log` & `system_resources.json`: Detailed system resource report covering active parameters, parameter savings, mean/min/max/total training time, memory usage (RSS and Apple Silicon MPS allocated memory), internal state norms ($\|c_t\|_2$ and $\max n_t$), and per-sample inference latency across sequence lengths.
- `gate_activations_len_40.json`: Step-by-step trace of all 64 neurons across $T=40$ timesteps (`i_gate`, `f_gate`, `o_gate`, `z_t`, `c_state`, `n_state`, `h_state`, and the $T \times T$ effective attention matrix $\alpha_{t,k}$).

---

## 3. How to Run Experiments (`main.py`)

### 3.1 Environment Setup
Using [uv](https://github.com/astral-sh/uv) or a standard Python 3.10+ virtual environment:
```bash
uv sync
```

### 3.2 CLI Usage & Flags
`main.py` provides a clean flag-based CLI to run any combination of tasks, forget-gate configurations, and execution modes:

```bash
python main.py \
  --task [parity | even_pairs | custom_cycle_nav | all] \
  --forget_gate [enabled | disabled | all] \
  --mode [train_and_eval | eval_only | plot_only]
```

#### Common Commands

1. **Run all tasks for both enabled and disabled forget gates (full training + OOD evaluation + figures):**
   ```bash
   python main.py --task all --forget_gate all --mode train_and_eval
   ```

2. **Run a single task (e.g., Parity Check) for both variants:**
   ```bash
   python main.py --task parity --forget_gate all --mode train_and_eval
   ```

3. **Run only the Forget Gate Disabled (`f_t = 1.0`) ablation on Custom Cycle Navigation:**
   ```bash
   python main.py --task custom_cycle_nav --forget_gate disabled --mode train_and_eval
   ```

4. **Re-evaluate saved checkpoints (`model_best.pth`), benchmark inference speed/resources, and regenerate all plots without retraining:**
   ```bash
   python main.py --task all --forget_gate all --mode eval_only
   ```

5. **Regenerate all figures and `resource_comparison.log` tables from saved JSON logs:**
   ```bash
   python main.py --task all --forget_gate all --mode plot_only
   ```

#### Optional CLI Flags
- `--epochs <int>`: Override the default number of epochs (`100` for Parity Check and Even Pairs; `150` for Custom Cycle Navigation).
- `--batch_size <int>`: Override batch size (`128` for Parity Check and Even Pairs; `256` for Custom Cycle Navigation).
- `--lr <float>`: Override learning rate (`0.002` for Parity Check and Even Pairs; `0.0012` for Custom Cycle Navigation).
- `--seed <int>`: Random seed (default: `42`).
- `--device [auto | mps | cuda | cpu]`: Compute device selection (default: `auto`).
- `--comet_online`: Stream metrics live to Comet ML online in addition to writing offline Comet archives and `comet_carbon_copy.jsonl`.

---

## 4. Logging, Comet ML & System Resource Profiling

Every experiment run records two parallel telemetry streams:
1. **Comet ML + Local Carbon Copy (`comet_carbon_copy.jsonl` & `train.log`)**:
   - Logs all hyperparameters, parameter counts, per-epoch losses/accuracies, validation accuracies at $L \in \{20, 40, 80, 120\}$, OOD accuracy curves, and resource metrics to Comet ML (`project_name="xlstm-forget-gate-study"`) while simultaneously appending structured JSON-Lines records to `<task>/<variant>/comet_carbon_copy.jsonl`.
2. **System Resources & Efficiency Profiler (`system_resources.log` & `resource_comparison.log`)**:
   - Tracks active forward parameters vs. eliminated forget-gate parameters ($W_f, R_f, b_f$: $8,256$ parameters across 4 heads), mean and minimum epoch duration, total training time, training throughput (samples/sec), peak RSS memory, MPS/GPU tensor memory, cell/normalizer state norms ($\|c_t\|_2, \max n_t$), and batch-32 inference latency across $L \in \{20, 40, 80, 120, 256\}$.

---

## 5. Consolidated Experimental Results

### Table 1: Accuracy, Length Generalization & Internal State Dynamics (Decoupled Stabilizer $m_t = 0$)

| Task | Forget Gate Variant | Active Params | Param Delta | Min Train Loss | Train Acc | ID Val Acc ($L=20$) | ID Val Acc ($L=40$) | OOD Acc ($L=80$) | OOD Acc ($L=120$) | OOD Acc ($L=256$) | Mean $\|c_t\|_2$ ($L=40$) | Max $n_t$ ($L=40$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parity Check** | **Enabled** ($f_t = \sigma(\tilde{f}_t)$) | 58,560 | Baseline | 0.00006 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 136.02 | 711.51 |
| **Parity Check** | **Disabled** ($f_t \equiv 1.0$) | **50,304** | **-14.10%** (-8,256) | 0.00011 | **100.0%** | **100.0%** | **100.0%** | 55.2% | 54.8% | 44.8% | 2,528.80 | 42,505.20 |
| **Even Pairs** | **Enabled** ($f_t = \sigma(\tilde{f}_t)$) | 58,560 | Baseline | 0.00003 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 6.69 | 6.44 |
| **Even Pairs** | **Disabled** ($f_t \equiv 1.0$) | **50,304** | **-14.10%** (-8,256) | 0.00001 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 76.43 | 134.97 |
| **Custom Cycle Nav** | **Enabled** ($f_t = \sigma(\tilde{f}_t)$) | 59,584 | Baseline | 0.00016 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 37.84 | 70.62 |
| **Custom Cycle Nav** | **Disabled** ($f_t \equiv 1.0$) | **51,328** | **-13.86%** (-8,256) | 0.00338 | **99.9%** | **100.0%** | **99.6%** | **100.0%** | 79.6% | 32.0% | 440.93 | 945.86 |

> **Note on Extreme OOD for Even Pairs**: At $L=400$ and $L=500$, **Even Pairs (Disabled, $f_t \equiv 1.0$)** maintains **100.0% accuracy** ($L=400: 100.0\%, L=500: 100.0\%$), slightly outperforming **Even Pairs (Enabled, $f_t = \sigma(\tilde{f}_t)$)** ($L=400: 99.6\%, L=500: 98.0\%$) because $f_t \equiv 1.0$ guarantees zero exponential decay of the initial boundary token $x_0$ over arbitrary horizons.

---

### Table 2: System Resources, Training Speed & Inference Latency Comparison

| Task | Forget Gate Variant | Active Params | Mean Epoch Time | Min Epoch Time | Total Train Time | Train Throughput | Peak RSS Memory | Inference $L=20$ | Inference $L=40$ | Inference $L=80$ | Inference $L=120$ | Inference $L=256$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parity Check** (100 ep, $N=4\text{k}$) | **Enabled** ($f_t = \sigma$) | 58,560 | 0.9245 s | 0.8901 s | 92.45 s | 4,326.8 seq/s | 620.97 MB | 0.0521 ms | 0.0911 ms | 0.1747 ms | 0.2534 ms | 0.5306 ms |
| **Parity Check** (100 ep, $N=4\text{k}$) | **Disabled** ($f_t \equiv 1.0$) | **50,304** (-14.1%) | 1.0098 s | 0.9679 s | 100.98 s | 3,961.2 seq/s | **589.31 MB** | 0.0559 ms | 0.0974 ms | 0.1898 ms | 0.2743 ms | 0.5984 ms |
| **Even Pairs** (100 ep, $N=4\text{k}$) | **Enabled** ($f_t = \sigma$) | 58,560 | 1.0710 s | 1.0710 s | 107.10 s | 3,734.8 seq/s | 636.22 MB | 0.0499 ms | 0.0861 ms | 0.1753 ms | 0.2500 ms | 0.5133 ms |
| **Even Pairs** (100 ep, $N=4\text{k}$) | **Disabled** ($f_t \equiv 1.0$) | **50,304** (-14.1%) | **1.0450 s** | **1.0450 s** | **104.50 s** (-2.4%) | **3,827.8 seq/s** | 636.22 MB | 0.0555 ms | 0.0968 ms | 0.1777 ms | 0.2630 ms | 0.6022 ms |
| **Custom Cycle Nav** (150 ep, $N=100\text{k}$) | **Enabled** ($f_t = \sigma$) | 59,584 | 27.2517 s | 26.8983 s | 4,087.76 s | 3,669.5 seq/s | 493.44 MB | 0.0498 ms | 0.0873 ms | 0.1662 ms | 0.2531 ms | 0.5182 ms |
| **Custom Cycle Nav** (150 ep, $N=100\text{k}$) | **Disabled** ($f_t \equiv 1.0$) | **51,328** (-13.9%) | **27.0408 s** | **26.8813 s** | **4,056.12 s** (-31.6s) | **3,698.1 seq/s** | 526.55 MB | 0.0538 ms | 0.0955 ms | 0.1770 ms | 0.2626 ms | 0.5605 ms |

### Key Takeaways from the Consolidated Results
1. **Parameter Efficiency**: Removing the forget gate ($W_f, R_f, b_f$) reduces total model parameters by **14.10%** on binary tasks (`58,560 -> 50,304`) and **13.86%** on 5-symbol Custom Cycle Navigation (`59,584 -> 51,328`), and reduces recurrent sLSTM cell parameters by **25.0%**.
2. **When Removing the Forget Gate is Better and Faster (`Even Pairs`)**: For long-range boundary comparison (`even_pairs`), removing the forget gate reduces training time by **2.4%** (`104.50s` vs. `107.10s`), saves **14.10%** of parameters, and improves extreme OOD generalization at $L=500$ from **98.0% to 100.0%** because the initial token memory is never attenuated by multiplicative forgetting.
3. **Why the Forget Gate is Essential for Unbounded State Tracking (`Parity Check` & `Custom Cycle Navigation`)**: While exponential input-gate scaling ($i_t = \exp(\tilde{i}_t)$) allows the Forget Gate Disabled model to solve in-distribution sequences ($L \le 40$) at **99.6%–100.0%** accuracy and even generalize up to $L=80$ (**100.0%**) on Custom Cycle Navigation, the lack of a multiplicative reset causes the normalizer state $n_t$ to grow by orders of magnitude ($\max n_t = 42,505.2$ on Parity Check and $945.9$ on Custom Cycle Navigation at $L=40$, compared to $711.5$ and $70.6$ with the forget gate enabled). Beyond $L \ge 80\text{--}120$, this normalizer saturation prevents further state transitions, confirming that the forget gate in decoupled sLSTM acts primarily as a **dynamic normalizer reset mechanism** rather than a simple memory erasure gate.
