# Reproducing the paper

All commands below run from the repository root. All intervals count optimizer updates. The software, hardware, batching, and numerical precision used for the paper are recorded below.

## Software and hardware

| Component | Version / configuration |
|---|---|
| GPU | 4 × NVIDIA GH200 120GB, one node |
| Python | 3.10.19 |
| PyTorch | 2.8.0a0+gitba56102 |
| CUDA | PyTorch build: 12.6; available nvcc toolkit: 12.9.41 |
| vLLM | 0.11.0 |
| Math-Verify | 0.9.0 |
| Transformers | 4.57.6 |
| Flash-Attention | 2.8.3 |
| Ray | 2.53.0 |
| verl | 0.9.0.dev0, with the modifications included in this repository |

## Parallelism and batching

Actor and reference models use FSDP across four GPUs. vLLM uses tensor parallelism 1 and pipeline parallelism 1; sequence parallelism is 1. Actor parameters and optimizer state are offloaded to CPU. Gradient checkpointing and padding removal are enabled.

Each optimizer update uses 32 prompts × 8 responses = 256 response sequences. The actor microbatch contains 8 response sequences per GPU, giving 8 gradient-accumulation microbatches per optimizer update on four GPUs. Actor dynamic batching is disabled. The collection multiplier `N` controls the number of optimizer updates per behavior collection; it is separate from gradient accumulation.

For GRPO+TIS, old-learner log-probability computation uses a microbatch of 1 response per GPU. The training launcher enables reference dynamic batching with a 16,384-token budget per GPU by default. Rollout uses a 4,096-token context limit, at most 8,192 batched tokens and 256 concurrent sequences, with GPU memory utilization set to 0.6.

## Numerical precision

Actor and reference parameters are stored in FP32, forward passes use BF16 mixed precision, and gradient reduction uses FP32. vLLM rollouts use BF16. For both GGPO and AGRO, learner and reference token log-probabilities are cast to FP32 before masked summation over response tokens. Adjusted rewards and the GGPO log-sum-exp are evaluated in FP32. This does not imply that the preceding forward pass or token-level log-probability kernel runs entirely in FP32.

## Installation

Start from a Linux aarch64 environment with the GPU software builds above already installed. The recorded PyTorch build is a custom build, not the standard PyPI wheel.

```bash
conda activate "${CONDA_ENV_NAME:?Set CONDA_ENV_NAME}"
python -m pip install --no-deps -r examples/ggpo/requirements-reproduction.txt
python -m pip install --no-deps -e .
```

## Environment notes

The pinned dependency file records installed versions, not a resolver-generated lock: vLLM declares `torch==2.8.0`, whereas the experiments use the custom PyTorch build listed above; pyvers declares `packaging<26`, whereas the runtime imports Packaging 26.0. Use `--no-deps` to preserve this environment. Optional TensorBoard logging was not installed or used; the launchers default to console logging.

## Data preparation

Use the two training dataset revisions listed in Appendix B of the paper:

```bash
export DATA_ROOT=/path/to/ggpo-data
export GURU_REVISION='2e2790a962a3c099bfb5ea61389cbf98a5ea439b'
export OPENR1_REVISION='da327b36b4dc077240aa8efd7f25c4017cd74ef4'
python -m examples.data_preprocess.alp_math \
  --output-dir "$DATA_ROOT" \
  --guru-revision "$GURU_REVISION" \
  --openr1-revision "$OPENR1_REVISION"
```

This creates `train.parquet`, `math500.parquet`, `aime24.parquet`, `aime25.parquet`, `olympiadbench.parquet`, and `minerva_math.parquet`. Rewards use binary Math-Verify correctness against the ground truth. Logging defaults to the console; set `WANDB_MODE=online` to use an existing W&B login.

The preprocessing code pins the following evaluation dataset snapshots for public reproduction, selected on September 24, 2026. Historical evaluation download revisions have not been verified.

| Dataset | Revision |
|---|---|
| HuggingFaceH4/MATH-500 | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| math-ai/aime24 | `83a7f387baaa524a8bda0022eac0541582297103` |
| math-ai/aime25 | `563bb8404243c5f09de6ec262f2db674fe5bce9b` |
| math-ai/olympiadbench | `4faaf1e6ec17d11a4218a9bf4c049ecaf954dd84` |
| math-ai/minervamath | `ee46ddc498933b1977577953250ca5c66be64f96` |

## Method

For each prompt $x_g$, sample $m$ responses from a behavior policy. Let $r_{g,i}$ be the verifier reward and $\pi_{\mathrm{ref}}$ the reference policy. Define the adjusted rewards and within-group weights:

```math
\begin{aligned}
R_{g,i} &= r_{g,i}-\beta\log\frac{\pi_\theta(a_{g,i}\mid x_g)}{\pi_{\mathrm{ref}}(a_{g,i}\mid x_g)},\\
q_{g,i} &= \frac{\exp(R_{g,i}/\beta)}{\sum_{j=1}^{m}\exp(R_{g,j}/\beta)}.
\end{aligned}
```

GGPO minimizes the log-mean-exp gap. For a minibatch of $G$ prompt groups and mean response length $\bar\ell$, the normalized training loss is:

```math
\begin{aligned}
\widehat{\mathcal J}_g
&=\beta\log\left(\frac{1}{m}\sum_{i=1}^{m}e^{R_{g,i}/\beta}\right)
  -\frac{1}{m}\sum_{i=1}^{m}R_{g,i},\\
L_{\mathrm{GGPO}}
&=\frac{1}{G\beta\bar\ell\sqrt{m-1}}\sum_{g=1}^{G}\widehat{\mathcal J}_g.
\end{aligned}
```

Holding rewards, sampled tokens, and reference probabilities fixed during each update gives:

```math
\nabla_\theta L_{\mathrm{GGPO}}
=-\frac{1}{T}\sum_{g=1}^{G}\sum_{i=1}^{m}
\frac{m q_{g,i}-1}{\sqrt{m-1}}
\nabla_\theta\log\pi_\theta(a_{g,i}\mid x_g),
\qquad T=Gm\bar\ell.
```

Each sequence log probability is the sum over its valid response tokens; $T$ is the total valid response-token count in the minibatch. The implementation differentiates the group loss directly and uses no learner-to-behavior importance weight, PPO clipping, additive KL loss, or entropy bonus. Reference probabilities are held fixed between scheduled refreshes. The reward KL term is enabled by default; its ablation switch is documented below.

## Training settings

All training commands below enable reference dynamic batching with a 16,384-token budget per GPU by default, including fixed-pool offline training.

| Setting | Default |
|---|---|
| Model | Qwen3-1.7B-Base (or Qwen3-4B-Base) |
| Optimizer | AdamW, learning rate 1e-6, betas (0.9, 0.999), weight decay 0.01 |
| Schedule | Constant, no warmup, gradient norm clip 1.0 |
| Learner batch | 32 prompt groups, 8 responses per group, one epoch |
| Collection | 32N groups, N=8; consume disjoint minibatches once |
| Training budget | 8,000 updates with periodic behavior synchronization; 400 fixed-pool offline |
| Evaluation | Every 160 updates with periodic behavior synchronization; every 80 fixed-pool offline |
| Checkpoints | Every 160 updates |
| Reference | Every 160 updates for GGPO/AGRO; fixed initial model for GRPO |
| β / GRPO KL coefficient | 0.01 |
| GRPO | Sample-standard-deviation advantages (epsilon 1e-6), token ratio clip [0.8, 1.2], dual clip 10 |
| GRPO+TIS | Compute old learner log probabilities once per collection; freeze them and the token-level TIS weights across all N updates; cap weights at 2; no batch normalization |
| Length limits | 2,048 prompt tokens; 2,048 response tokens |
| Sampling | Temperature 1, top-p 1, top-k disabled; 32 evaluation responses |
| Reward filtering | Keep complete groups with at least one correct and one incorrect response; refill up to 32 batches |
| Entropy bonus | None |

GGPO uses sequence sums and direct autograd through the group objective, with the paper's global response-length normalization. AGRO uses sequence sums and detached leave-one-out advantages, without an extra exterior β factor. Neither uses PPO ratios, clipping, or an additive KL loss. Reference refresh leaves the already-collected responses and behavior log probabilities unchanged.

GGPO includes the reference KL term in its group weights by default (`GGPO_REWARD_KL=true`). For the reward-only ablation, use `GGPO_REWARD_KL=false ALGORITHM=ggpo bash examples/ggpo/run_math.sh` (also supported with `MODE=offline`). The default run name and checkpoint directory gain a `-reward-only` suffix; an explicit `RUN_NAME` or `CKPTS_DIR` takes precedence. This uses `q = softmax(r / β)` with detached coefficients `a_i = (m q_i - 1) / sqrt(m - 1)` and minimizes the surrogate `-sum_i a_i log πθ(y_i|x_i) / T`, where `T` is the global valid response-token count. β remains the softmax temperature; sequence/token normalization is unchanged. In this mode, the logged GGPO loss is the policy-gradient surrogate. Reference log-probabilities remain available for diagnostics.

## Default setting and ablations

```bash
# Default comparisons: repeat each method at seeds 42, 1234, 2027 for 1.7B
ALGORITHM=ggpo SEED=42 bash examples/ggpo/run_math.sh
ALGORITHM=agro SEED=42 bash examples/ggpo/run_math.sh
ALGORITHM=grpo SEED=42 bash examples/ggpo/run_math.sh

# 4B: seed 42; use each method above
MODEL_PATH=Qwen/Qwen3-4B-Base ALGORITHM=ggpo SEED=42 bash examples/ggpo/run_math.sh

# Staleness: N in {8,16,32,64,128}, default model and β
ALGORITHM=ggpo N=128 bash examples/ggpo/run_math.sh
ALGORITHM=grpo-tis N=128 bash examples/ggpo/run_math.sh

# Group size: 4, 8, or 16; retain 32 prompt groups per optimizer update
ALGORITHM=ggpo GROUP_SIZE=16 bash examples/ggpo/run_math.sh

# GGPO/AGRO reference ablation: 0 (fixed), 160, or 320
ALGORITHM=ggpo REFRESH_EVERY=0 bash examples/ggpo/run_math.sh
ALGORITHM=ggpo REFRESH_EVERY=160 BETA=0.1 bash examples/ggpo/run_math.sh

# GRPO reference ablation: fixed by default
ALGORITHM=grpo GRPO_REFRESH_EVERY=320 bash examples/ggpo/run_math.sh
```

The N=128 condition uses seeds 42, 1234, and 2027; other non-default ablations and 4B use seed 42. At N=128, the last collection is generated in full and only its first 64 minibatches are consumed before update 8,000. Evaluation or reference refresh within a collection does not regenerate its remaining samples. Checkpoints preserve the unconsumed collection for resume.

For GRPO+TIS, the PPO ratio is current learner / frozen old learner, while the TIS weight is min(old learner / rollout, 2). Both old log probabilities and TIS weights are computed before the first update on the collection and saved with its remaining minibatches, including across validation and checkpoint resume.

## Fixed-pool offline training

```bash
export POOL_DIR=/path/to/pool-seed42
MODE=collect SEED=42 bash examples/ggpo/run_math.sh
MODE=offline SEED=42 ALGORITHM=grpo bash examples/ggpo/run_math.sh
MODE=offline SEED=42 ALGORITHM=grpo-tis bash examples/ggpo/run_math.sh
MODE=offline SEED=42 ALGORITHM=agro bash examples/ggpo/run_math.sh
MODE=offline SEED=42 ALGORITHM=ggpo bash examples/ggpo/run_math.sh
```

Use a new pool directory for each seed and share it across the four methods. The collector keeps the initial Qwen3-1.7B-Base model fixed, applies the same mixed-reward filtering, and stores response tokens, masks, rewards, and behavior log probabilities. Collection uses `N=1` to save 400 individual learner minibatches. Offline training loads eight consecutive minibatches at a time (`N=8`) and consumes each once in order; rollout workers only serve evaluation. GRPO+TIS computes old learner probabilities and capped token TIS weights once for the whole block before its first update, then holds them fixed across all eight updates. The collector's behavior log probabilities remain fixed throughout training. Checkpoint resume preserves both the unconsumed block and these frozen probabilities.

Offline mismatch diagnostics also aggregate eight updates, covering 2,048 responses at the default group size. Use `offline_block/rollout_training_mismatch/log_ratio_abs_mean` for token-weighted drift and `offline_block/rollout_training_mismatch/raw_is/ess_normalized` for normalized ESS. ESS is recomputed from the pooled sequence log-ratios, not averaged across minibatches. The partial diagnostic accumulator is checkpointed with the remaining block; ordinary per-update diagnostics remain available under `rollout_training_mismatch/`.

## Checkpoints and final evaluation

Training checkpoints include actor/optimizer state, dataloader state, moving reference state where applicable, and any remaining collection. The trainer's normal resume mechanism is available through trailing Hydra overrides, for example `trainer.resume_mode=resume_path trainer.resume_from_path=/path/to/global_step_160`.

Export an actor checkpoint to Hugging Face format, then evaluate it on MATH-500 (500), AIME 2024 (30), AIME 2025 (30), OlympiadBench (674), and Minerva Math (272):

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir /path/to/global_step_8000/actor --target_dir /path/to/exported-model
MODEL_PATH=/path/to/exported-model EVAL_DIR=/path/to/evaluation \
  bash examples/ggpo/evaluate.sh
python examples/ggpo/summarize_eval.py /path/to/evaluation/0.jsonl
```

`pass@1` in the summary is mean@32. The overall average weights all 1,506 problems equally, rather than averaging the five benchmark percentages. For the default setting and ablations with periodic behavior synchronization, checkpoint selection uses the final 8,000-update model. Offline trajectories use MATH-500 evaluations at updates 0, 80, 160, 240, 320, and 400.

## Configuration and CPU checks

`MODEL_PATH`, `SEED`, `N`, `GROUP_SIZE`, `BETA`, `REFRESH_EVERY`, and `GRPO_REFRESH_EVERY` select paper settings. `NGPUS_PER_NODE` (default 4), `NNODES`, `PPO_MICRO_BATCH_SIZE_PER_GPU`, `ROLLOUT_TP`, `ROLLOUT_GPU_MEMORY_UTILIZATION`, `ACTOR_PARAM_OFFLOAD`, and `ACTOR_OPTIMIZER_OFFLOAD` adapt execution to available hardware. Model/tokenizer revisions and exact evaluation sampling RNG state are not specified in the paper. Use the desired local model snapshot when controlling those details.

```bash
DRY_RUN=1 ALGORITHM=ggpo N=128 bash examples/ggpo/run_math.sh
python -m pytest -q tests/examples/test_ggpo_launchers_on_cpu.py
```

These CPU checks exercise configuration composition and collection/offline bookkeeping. They do not replace a GPU training smoke run on the target CUDA/vLLM environment.
