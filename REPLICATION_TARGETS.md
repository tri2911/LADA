# LADA — Replication Targets

What outputs we must reproduce from **"LADA: Scalable Label-Specific CLIP Adapter for Continual Learning"** (Luo et al., ICML 2025), and how each maps onto this repo.

This is the checklist: every table and figure in the paper, the headline numbers to hit, and whether this repo can produce it directly.

---

## 0. Setup recap (so the numbers are comparable)

- **Benchmark:** X-TAIL (Cross-domain Task-Agnostic Incremental Learning), 10 datasets / 10 tasks, 1,100 classes total.
  Datasets: Aircraft, Caltech101, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397.
- **Backbone:** CLIP ViT-B/16. Optimizer AdamW, lr 1e-3, batch 64. Hyperparams **λ₁ = 16** (LADA dim), **λ₂ = 4** (prototypes/class).
- **Two shot regimes:** 16-shot and full-shot.
- **Two task orders:**
  - **Order-I** (alphabetical): Aircraft → Caltech101 → DTD → EuroSAT → Flowers → Food → MNIST → OxfordPet → StanfordCars → SUN397. (`configs/data/TAIL.yaml`)
  - **Order-II** (random): StanfordCars → Aircraft → OxfordPet → Food → SUN397 → MNIST → Flowers → DTD → Caltech101 → EuroSAT. (`configs/data/TAIL_order2.yaml`)
- **Three metrics** (defined Appendix A; `result_process.py` computes them). `â_k^(j)` = accuracy on task *k* after training on task *j*:
  - **Transfer**ₖ = mean over j=1..k−1 of `â_k^(j)` → forward transfer / forgetting of pre-trained knowledge on *future* tasks. (no value for task 1)
  - **Average**ₖ = mean over j=1..K of `â_k^(j)` → holistic stability+plasticity.
  - **Last**ₖ = `â_k^(K)` → final accuracy after all training (backward forgetting).
  - Reported "Mean" = average of the per-task metric across the 10 tasks.

The repo's pipeline (`run_TAIL_*.sh` → `result_process.py`) emits, **per setting**, the full 10×10 accuracy matrix plus the Transfer/Average/Last row and their means. That matrix *is* Tables 7–10, and its means are the "Ours" entries of Tables 1/2/5/6.

---

## 1. PRIMARY headline numbers (LADA / "Ours")

These four runs are the core of the replication — fully supported by this repo. Match the means within ~0.1–0.3.

| Setting | Script | Transfer | Average | Last |
|---|---|---|---|---|
| 16-shot, Order-I | `scripts/run_TAIL_16shot.sh` | **61.5** | **72.7** | **83.1** |
| Full-shot, Order-I | `scripts/run_TAIL_fullshot.sh` | **61.9** | **75.2** | **86.9** |
| 16-shot, Order-II | `scripts/run_TAIL_16shot_order2.sh` | **56.7** | **68.9** | **83.3** |
| Full-shot, Order-II | `scripts/run_TAIL_fullshot_order2.sh` | **55.4** | **69.2** | **86.9** |

(Zero-shot CLIP baseline reference, same for all settings: Transfer/Average context = **57.7** mean accuracy. `scripts/run_TAIL_zeroshot.sh`.)

### Chosen plan (decided 2026-05-23)
**Must-have scope = 16-shot Order-I only** (cheapest, canonical; Tables 1 & 7). Two-notebook Colab workflow:
- **`lada_colab.ipynb` (train):** runs the 10-task chain, snapshots `checkpoint.pth.tar` after each task to Drive as `runs/TAIL_16shot_orderI/checkpoints/stepNN_<ds>.pth.tar`. Resumable across disconnects. Zero repo edits (config already λ₁=16, λ₂=4, `continue_train=True`).
- **`lada_colab_eval.ipynb` (eval):** loads each step checkpoint, runs `test_wo_selector` (→ Table 7 row + Figure 2 curve) and `test_w_selector` (→ Figure 3), saves per-sample **argmax preds + task routing** to `predictions/stepNN_{wo,w}.npz`, reconstructs Table 7, and plots Fig 2 / Fig 3.

This single setting yields: **Table 7 ⇒ Table 1 "Ours" row + headline means (61.5/72.7/83.1)**, the **16-shot Figure 2 LADA curve**, and the **16-shot panel of Figure 3** (task recall is already computed by the evaluator). Baseline rows and the full-shot/Order-II tables remain out of scope unless revisited.

---

## 2. TABLES

### Table 1 — X-TAIL 16-shot, Order-I (per-dataset Transfer/Average/Last)
Methods: Zero-shot, LwF, WiSE-FT, ZSCL, MoE-Adapters, Primal-RAIL, Dual-RAIL, **Ours**.
- **Repo support:** "Ours" row + Zero-shot only. Baselines need their own repos.
- **Ours target row** (Aircraft, Caltech101, DTD, EuroSAT, Flowers, Food, MNIST, Pets, Cars, SUN397 | Mean):
  - Transfer: –, 75.0, 36.1, 35.9, 66.3, 83.7, 42.1, 88.0, 65.3, 61.4 | **61.5**
  - Average: 49.1, 91.0, 61.3, 71.6, 84.4, 85.0, 62.8, 89.7, 69.2, 62.9 | **72.7**
  - Last: 49.6, 93.7, 69.3, 86.9, 96.7, 86.9, 93.8, 93.7, 84.6, 76.0 | **83.1**
- Baseline means to rank against (Transfer / Average / Last): LwF 47.7/53.2/62.8 · WiSE-FT 52.3/54.2/58.0 · ZSCL 59.0/60.0/63.4 · MoE-Adapters 56.0/63.0/70.5 · Primal-RAIL –/71.1/81.4 · Dual-RAIL –/71.3/82.3.

### Table 2 — X-TAIL full-shot, Order-I
Same structure as Table 1 (Dual-RAIL omitted — can't finish full-shot under memory budget).
- **Repo support:** "Ours" row + Zero-shot.
- **Ours target row:**
  - Transfer: –, 75.2, 36.1, 36.7, 65.6, 83.9, 45.2, 88.0, 65.3, 61.1 | **61.9**
  - Average: 53.9, 93.6, 66.6, 78.0, 85.3, 86.7, 65.2, 89.9, 69.7, 62.7 | **75.2**
  - Last: 55.5, 96.2, 75.8, 95.8, 98.4, 89.6, 98.8, 94.5, 87.3, 77.2 | **86.9**

### Table 3 — Ablation: BF / DPT / LADA (16-shot & full-shot, Order-I)
Component ablation. Rows = enabling {BF}, {BF+DPT}, {BF+DPT+LADA}.
- **Repo support:** Needs config toggles to disable DPT and LADA modules (verify flags exist in `main.py`/`trainer.py`).
- Targets (Transfer / Average / Last):
  - 16-shot: BF 59.4/70.9/82.1 · +DPT 59.9/71.6/82.6 · +LADA 61.2/72.3/83.0 · **full 61.5/72.7/83.1**
  - Full-shot: BF 59.6/73.2/85.6 · +DPT 60.3/74.4/86.3 · +LADA 61.1/74.9/86.8 · **full 61.9/75.2/86.9**

### Table 4 — λ₁ (LADA dim) × λ₂ (prototypes) sweep, full-shot
Grid: λ₁ ∈ {8, 16, 32*}, λ₂ ∈ {1, 4, 16}. Reports Transfer/Average/Last **plus Time (s/batch), Memory (GB), Params (M)**.
- **Repo support:** Needs to set λ₁/λ₂ via CLI/config, and to instrument **per-batch time, peak GPU memory, and LADA param count**. (32* = some classes have <32 samples.)
- Default cell (λ₁=16, λ₂=4) target: 61.9 / 75.2 / 86.9, Time 0.289, Mem 18.51 GB, Params 9.01 M.
- Full grid (for completeness):
  | λ₁ | λ₂ | Transfer | Avg | Last | Time | Mem | Params |
  |----|----|----|----|----|----|----|----|
  | 8 | 1 | 60.9 | 74.7 | 86.7 | 0.280 | 17.84 | 4.51 |
  | 8 | 4 | 61.7 | 75.1 | 86.7 | 0.287 | 18.14 | 4.51 |
  | 8 | 16 | 62.2 | 75.1 | 86.7 | 0.318 | 19.05 | 4.51 |
  | 16 | 1 | 61.4 | 75.1 | 86.9 | 0.281 | 18.01 | 9.01 |
  | 16 | 4 | 61.9 | 75.2 | 86.9 | 0.289 | 18.51 | 9.01 |
  | 16 | 16 | 62.3 | 74.8 | 86.9 | 0.330 | 20.62 | 9.01 |
  | 32* | 1 | 60.9 | 74.7 | 86.9 | 0.282 | 18.42 | 17.51 |
  | 32* | 4 | 61.2 | 74.6 | 86.9 | 0.297 | 18.97 | 17.51 |
  | 32* | 16 | 61.9 | 74.0 | 86.9 | 0.358 | 23.13 | 17.51 |

  > Note: Time/Memory targets are from a single NVIDIA 4090. Our shared RTX 5070 Ti with GPU≤16GB / RAM≤20GB caps means absolute Time/Memory will differ — treat accuracy as the match target and Time/Memory/Params as trend-only.

### Table 5 — X-TAIL 16-shot, Order-II
Same structure as Table 1, Order-II column ordering (Cars, Aircraft, Pets, Food, SUN397, MNIST, Flowers, DTD, Caltech101, EuroSAT).
- **Ours target row:**
  - Transfer: –, 23.8, 87.8, 84.3, 61.3, 42.2, 65.7, 37.2, 71.9, 36.0 | **56.7**
  - Average: 84.7, 46.2, 92.2, 86.4, 70.0, 68.0, 77.4, 47.2, 75.9, 41.2 | **68.9**
  - Last: 84.8, 49.2, 93.6, 87.6, 76.6, 93.9, 97.4, 70.7, 91.7, 87.4 | **83.3**

### Table 6 — X-TAIL full-shot, Order-II
- **Ours target row:**
  - Transfer: –, 23.8, 87.8, 84.3, 61.4, 41.6, 65.7, 34.5, 71.0, 28.9 | **55.4**
  - Average: 87.0, 49.3, 93.1, 88.0, 70.9, 66.8, 79.0, 46.8, 76.0, 35.6 | **69.2**
  - Last: 87.0, 55.4, 94.5, 89.6, 77.7, 98.8, 99.0, 75.8, 95.8, 86.9 | **86.9**

### Tables 7–10 — Full per-step accuracy matrices for "Ours" (the appendix dumps)
These are exactly what `result_process.py` prints. **100% reproducible from this repo** and the most direct verification target.
- Table 7: 16-shot Order-I — 10×10 matrix + Transfer/Avg/Last (means 61.5 / 72.7 / 83.1).
- Table 8: full-shot Order-I — (means 61.9 / 75.2 / 86.9).
- Table 9: 16-shot Order-II — (means 56.7 / 68.9 / 83.3).
- Table 10: full-shot Order-II — (means 55.4 / 69.2 / 86.9).
- Diagonal = each task's accuracy right after training it; below-diagonal = retention (Last row = bottom); above-diagonal = forward transfer.

---

## 3. FIGURES / CHARTS

### Figure 1 — Conceptual paradigm diagram (prompt-based vs MoE-Adapters vs LADA)
- **Not a result.** No replication needed.

### Figure 2 — Accuracy (%) across all tasks over all learning steps, full-shot
- 10 small line charts (one per dataset). X-axis = learning step 0–10, Y-axis = accuracy. Lines: ZSCL, MoE-Adapters, RAIL, **LADA**.
- **Repo support:** The **LADA** line is fully reproducible — it's column-wise slices of the Table 8 matrix (each dataset's accuracy after each step). Baseline lines need other repos.
- **Action:** Write a plotting script that reads `output/<run>/result.txt` (or the raw `log_*.txt`) and draws the LADA curves; overlay baselines only if we obtain their logs.

### Figure 3 — Zero-shot CLIP as selector: with vs without selector (16-shot & full-shot)
- Two panels. Plots **task recall** (proportion of samples routed to correct task) and **accuracy difference** (LADA vs LADA-with-selector) per step. Shows LADA is better *without* a selector.
- **Repo support:** Needs extra instrumentation — a "with selector" variant (route via vanilla zero-shot CLIP first) and a **task-recall** metric, neither of which the default pipeline emits. **Lowest-priority / extra work.**

---

## 4. Reproducibility summary

| Output | Reproducible from this repo? | Notes |
|---|---|---|
| **§1 headline means (4 settings)** | ✅ Direct | The main goal. |
| Tables 7–10 (Ours matrices) | ✅ Direct | `result_process.py` output. |
| Tables 1/2/5/6 — *Ours* + Zero-shot rows | ✅ Direct | Baseline rows need external repos. |
| Tables 1/2/5/6 — baseline rows | ❌ | LwF, WiSE-FT, ZSCL, MoE-Adapters, RAIL not in repo. |
| Table 3 (ablation) | ⚠️ Toggles | Verify BF-only / no-DPT / no-LADA flags. |
| Table 4 (λ₁×λ₂ + cost) | ⚠️ Sweep + instrument | Need time/mem/param logging; Time/Mem hardware-dependent. |
| Figure 2 — LADA curves | ✅ Plot script | Baseline curves need external logs. |
| Figure 3 — selector study | ❌/⚠️ Extra | Needs selector variant + task-recall metric. |
| Figure 1 | n/a | Conceptual. |

### Recommended replication order
1. **Run the 4 LADA settings** and verify §1 means + Tables 7–10. (Core deliverable.)
2. Build a **plot script** for Figure 2 (LADA curves) from the result matrices.
3. **Table 3 ablation** via component toggles.
4. **Table 4 sweep** (λ₁, λ₂) + cost instrumentation.
5. (Optional) Baseline rows for Tables 1/2/5/6 and Figure 2 — only if reproducing the comparison is required.
6. (Optional) Figure 3 selector study — needs new metric code.
