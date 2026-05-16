# Dexora: Open-source VLA for High-DoF Bimanual Dexterity

> 📝 Paper: *Dexora: Open-source VLA for High-DoF Bimanual Dexterity* (ICRA'26 submission, `ICRA26_0209_FI.pdf` in this repo)
> 🌐 Project page: https://dexoravla.github.io
> 🤖 Hardware: 2× 6-DoF AIRBOT arms + 2× 12-DoF XHAND (36 DoF total)

Dexora is the first open-source Vision-Language-Action (VLA) system that natively targets **dual-arm, dual-hand, high-DoF** manipulation. It is built around three contributions:

1. **Hybrid teleoperation** that decouples gross arm kinematics (exoskeleton backpack) from fine finger motion (markerless Apple Vision Pro hand tracking), and that drives both a physical platform and an identical MuJoCo digital twin.
2. **Embodiment-matched corpus**: 100K simulated trajectories (6.5M frames) + 10K real teleoperated episodes (177.5h, 3.2M frames), all logged in the LIBERO-2.0 format.
3. **Discriminator-guided quality-aware training**: an offline discriminator scores each demonstration clip; the diffusion-transformer policy is post-trained with a weighted loss that down-weights low-quality demonstrations.

This repository contains the full **training**, **inference**, and **data-processing** code; large data and pretrained weights are released separately (see [Downloads](#downloads)).

---

## Repository Layout

```
Dexora/
├── configs/                # YAML configs
│   ├── base.yaml           #   1B policy (legacy)
│   ├── base_400m.yaml      #   400M paper spec (28 / 1024 / 16)
│   ├── scoring.yaml        #   30M discriminator
│   └── cross_embodiment/   #   EC-1 / EC-2 / EC-3 fine-tuning configs
├── models/
│   ├── rdt/                # Core diffusion-transformer blocks
│   ├── rdt_runner.py       # Stage-1/3 policy (supports weighted MSE loss, Eq.(8))
│   ├── scoring_model.py    # Stage-2 discriminator (Eq.(7))
│   └── sample_weighting.py # DWBC score → weight + Eq.(8) weighted MSE helper
├── train/
│   ├── train.py            # Stage-1: pretrain on simulation data
│   ├── train_scoring.py    # Stage-2: discriminator PU training
│   └── train_posttrain.py  # Stage-3: quality-aware post-training
├── data/                   # Dataset loaders (BSON, LeRobot, EgoDex, HDF5)
├── scripts/                # Eval & visualization (eval_smoothness.py, ...)
├── tools/
│   ├── release_check.py    # Release-readiness sanity checks (used by CI)
│   └── data/               # Internal data-ops scripts (see tools/data/README.md)
├── tests/                  # CPU-only pytest suite (DWBC, PU loss, EC configs, ...)
├── analyze_episode_quality.py  # Pre-screening (Aep + Jep, §III-C)
├── compute_logpi.py        # log-π proxy (Eq.(4)-(5))
├── replay_validate.py      # Spre → Shigh post-validation
├── main.py / main_scoring.py / main_posttrain.py    # Entry points
├── train_ours.sh / train_scoring.sh / post_train.sh # Stage launch scripts
├── run_all_stages.sh                                # End-to-end pipeline
├── Makefile                                         # `make help` for the menu
├── pyproject.toml + requirements{,-dev}.txt
├── .github/workflows/ci.yml                         # ruff + pytest + sanity + release-check
├── .pre-commit-config.yaml
├── LICENSE  +  CITATION.cff  +  CONTRIBUTING.md  +  CODE_OF_CONDUCT.md
└── ICRA26_0209_FI.pdf      # The paper
```

## Development quickstart

```bash
pip install -e ".[dev]"      # install runtime + dev deps (ruff, black, pytest, pre-commit)
pre-commit install            # auto-run lint/format on each commit
make test                     # 57 unit tests, ~3s, CPU-only
make lint                     # ruff + black --check
make release-check            # sanity: required files + README placeholders + script links
make all-stages               # run the entire 3-stage pipeline end-to-end
```

---

## Installation

```bash
# 1. Conda env
conda create -n dexora python=3.10 -y
conda activate dexora

# 2. PyTorch (CUDA 12.1 example; pick your own from pytorch.org)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Other deps
pip install packaging==24.0
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

## Downloads

Large assets are **not** included in this repo. Place them at the expected paths or update the configs.

### Pretrained foundation encoders (third-party)

| Asset | Approx. Size | Default Path | Source |
|---|---|---|---|
| SigLIP-SO400M vision encoder | ~3.7 GB | `google/siglip-so400m-patch14-384/` | [huggingface.co/google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) |
| T5-v1.1-XXL text encoder | ~44 GB | `google/t5-v1_1-xxl/` | [huggingface.co/google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) |

```bash
# One-time download (≈ 48 GB total). Requires `huggingface-cli login` if rate-limited.
huggingface-cli download google/siglip-so400m-patch14-384 \
    --local-dir google/siglip-so400m-patch14-384 --local-dir-use-symlinks False
huggingface-cli download google/t5-v1_1-xxl \
    --local-dir google/t5-v1_1-xxl --local-dir-use-symlinks False
```

### Dexora checkpoints & data (this work)

The official Dexora release is hosted on the project page; release links are published there as they become available:

> **🌐 Project page (canonical index):** https://dexoravla.github.io

| Asset | Approx. Size | Default Path | Status |
|---|---|---|---|
| Dexora 400M policy — Stage-1 (sim pretrain) | ~1.6 GB | `checkpoints/dexora-400m-pretrain/` | release pending — see project page |
| Dexora 400M policy — Stage-3 (post-trained on real) | ~1.6 GB | `checkpoints/dexora-400m-posttrain/` | release pending — see project page |
| Dexora discriminator (30M) | ~120 MB | `checkpoints/dexora-scoring/` | release pending — see project page |
| Synthetic corpus (100K trajectories, 6.5M frames, 361 h) | ~700 GB | `data/sim/` | release pending — see project page |
| Real corpus (10K episodes, 3.2M frames, 177.5 h, LIBERO-2.0 format) | ~3 TB | `data/real/` | release pending — see project page |

When the official URLs go live we will replace the *release pending* rows with direct
`huggingface.co/dexoravla/...` paths and `wget`-able mirror links.

### Upstream tooling referenced by the paper

The data-generation and benchmarking pieces of the pipeline rely on the following
open-source projects:

| Component | Used for | Link |
|---|---|---|
| LIBERO-2.0 | Real-data storage format | [github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| DexMimicGen | Synthetic trajectory synthesis (Sec III-B) | [github.com/NVlabs/DexMimicGen](https://github.com/NVlabs/DexMimicGen) |
| Objaverse / Objaverse-XL | Source of 3D assets for sim | [objaverse.allenai.org](https://objaverse.allenai.org/) |
| Qwen2.5-VL | VLM-driven asset mining & physical-property assignment | [huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) |
| MuJoCo | Digital twin & replay-based post-validation | [mujoco.org](https://mujoco.org) |
| LeRobot | Dataset loader / format reference | [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot) |
| RDT-1B | Architectural reference for the Diffusion-Transformer policy | [github.com/thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) |
| DWBC (Xu et al. ICML'22) | Score → weight mapping (Sec III-D, ref [41]) | [github.com/ryanxhr/DWBC](https://github.com/ryanxhr/DWBC) |

## Three-stage Training Recipe (Sec III-D)

### Stage 1 — Pretrain on simulation
Builds basic competence (pick-and-place, assemble, articulated objects).

```bash
bash train_ours.sh
# Uses configs/base_400m.yaml, --load_from="bson" (or "lerobot"),
# trains πθ for 100K steps on synthetic data.
```

### Stage 2a — Pre-screen real demonstrations (Aep, Jep)
Computes per-episode normalized acceleration / jerk RMS and keeps the bottom-20% intersection ≈ `Spre` (Sec III-C).

```bash
bash analyze_jerk.sh
# Writes new_lerobot_jerk/complete_analysis_results.json
```

### Stage 2b — Replay-based post-validation → `Shigh`
Open-loop replays each `Spre` episode in the MuJoCo digital twin to check **task completion** and **collision-free** execution. Only the survivors are used as positives.

```bash
python replay_validate.py \
    --pre_screening_file new_lerobot_jerk/complete_analysis_results.json \
    --output_file shigh_episodes.json
```

### Stage 2c — log-π proxy + discriminator training
Compute the policy-compatibility proxy `\hat{log π}` (Eq.(4)-(5), z-scored) using the Stage-1 policy, then train the discriminator with PU loss (Eq.(7), η=0.5).

```bash
bash compute_logpi.sh    # writes new_lerobot_logpi_values.json
bash train_scoring.sh    # writes checkpoints/scoring-model-v1
```

### Stage 3 — Data-quality-aware post-training (NEW)
Post-trains the Stage-1 policy on real data with **discriminator-weighted diffusion loss** (Eq.(8)):

$$\mathcal{L}_\pi = \sum_{i=1}^{L} w_i \, \lVert\varepsilon_\theta(\cdot) - \varepsilon\rVert_2^2$$

where $w_i = \text{DWBC}(d(\xi_i))$ is derived from the discriminator score via the DWBC mapping (with a short linear warm-up).

```bash
bash post_train.sh
# Loads:
#   - stage-1 policy (checkpoints/dexora-400m-pretrain/)
#   - stage-2 scoring model (checkpoints/scoring-model-v1/)
# Trains stage-3 with weighted MSE on real data.
```

## Inference / Deployment

```bash
# Real-robot rollout (see deploy/ for the rest)
python deploy/run_inference.py \
    --pretrained_model_name_or_path checkpoints/dexora-400m-posttrain/ \
    --config_path configs/base_400m.yaml
```

## Reproducing the Paper Numbers

| Table / Figure | How to run | Knob |
|---|---|---|
| Tab. I — Basic tasks (12) | Stage-1 + Stage-3 on each task; 20 rollouts | default |
| Tab. II — Dexterous tasks (6) | Same, on the 6 dexterous tasks | default |
| Tab. III — Discriminator ablation | Run Stage-3 with **and** without the discriminator; report S.R. + Acc + Jerk | `--no_quality_weights` (vanilla baseline) |
| Fig. 9, Tab. II EC rows — Cross-embodiment | Stage-3 ckpt + fine-tune under each EC config (see `configs/cross_embodiment/`) | `--config_path configs/cross_embodiment/{ec1_franka,ec2_aloha,ec3_g1_inspire}.yaml` |
| Fig. 10 — Data composition | Stage-3 with sim-only / sim+50%-real / sim+all-real | `--real_data_fraction {0.0, 0.5, 1.0}` |
| Fig. 11 — Per-joint trajectories | `scripts/eval_action_curves.py` after rollouts | — |
| Smoothness metrics (Acc.↓ / Jerk↓) | `scripts/eval_smoothness.py rollouts/*.json --stats_file new_lerobot_stats/dataset_statistics.json` | — |

## Citing

```bibtex
@inproceedings{dexora2026,
  title     = {Dexora: Open-source VLA for High-DoF Bimanual Dexterity},
  author    = {Zhang, Zongzheng and Pang, Jingrui and others},
  booktitle = {ICRA},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). Third-party components (SigLIP, T5, LeRobot, cutlass) keep their original licenses.
