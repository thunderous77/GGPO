#!/usr/bin/env bash
# Paper defaults. All step counts below are optimizer updates.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)
: "${DATA_ROOT:?Set DATA_ROOT to the prepared ALP-Math directory}"
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}
ALGORITHM=${ALGORITHM:-ggpo}
MODE=${MODE:-online}
BETA=${BETA:-0.01}
SEED=${SEED:-42}
GROUP_SIZE=${GROUP_SIZE:-8}
N=${N:-8}
TOTAL_UPDATES=${TOTAL_UPDATES:-8000}
EVAL_EVERY=${EVAL_EVERY:-160}
SAVE_EVERY=${SAVE_EVERY:-160}
REFRESH_EVERY=${REFRESH_EVERY:-160}
EXTRA=()
case "$MODE" in
    online) ;;
    collect | offline)
        : "${POOL_DIR:?Set POOL_DIR to the fixed rollout pool for this seed}"
        N=8
        TOTAL_UPDATES=400
        EVAL_EVERY=80
        EXTRA+=("+trainer.offline_pool_dir=${POOL_DIR}")
        if [[ "$MODE" == collect ]]; then
            N=1
            ALGORITHM=grpo
            EXTRA+=("+trainer.trainer_class=verl.trainer.ppo.paper_offline.CollectRollouts")
        else
            EXTRA+=("+trainer.trainer_class=verl.trainer.ppo.paper_offline.OfflineTrainer")
        fi
        ;;
    *) printf 'MODE must be online, collect, or offline\n' >&2; exit 2 ;;
esac
case "$ALGORITHM" in
    grpo | grpo-tis)
        LOSS=vanilla
        REFRESH_EVERY=${GRPO_REFRESH_EVERY:-0}
        EXTRA+=(
            "actor_rollout_ref.actor.use_kl_loss=true"
            "actor_rollout_ref.actor.kl_loss_coef=${BETA}"
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
            "actor_rollout_ref.actor.clip_ratio_low=0.2"
            "actor_rollout_ref.actor.clip_ratio_high=0.2"
            "actor_rollout_ref.actor.clip_ratio_c=10.0"
            "actor_rollout_ref.actor.loss_agg_mode=token-mean"
            "algorithm.rollout_correction.bypass_mode=true"
        )
        if [[ "$ALGORITHM" == grpo-tis ]]; then
            EXTRA+=("algorithm.rollout_correction.bypass_mode=false"
                "algorithm.rollout_correction.rollout_is=token"
                "algorithm.rollout_correction.rollout_is_threshold=2.0"
                "algorithm.rollout_correction.rollout_is_batch_normalize=false")
        fi
        ;;
    ggpo | agro)
        LOSS=$ALGORITHM
        EXTRA+=("actor_rollout_ref.actor.use_kl_loss=false"
            "actor_rollout_ref.actor.policy_loss.beta=${BETA}"
            "actor_rollout_ref.actor.policy_loss.sequence_log_prob_reduction=sum"
            "actor_rollout_ref.actor.policy_loss.memory_safe_surrogate=false")
        ;;
    *) printf 'ALGORITHM must be ggpo, agro, grpo, or grpo-tis\n' >&2; exit 2 ;;
esac
if [[ "$ALGORITHM" == ggpo ]]; then
    EXTRA+=("actor_rollout_ref.actor.policy_loss.reward_kl=${GGPO_REWARD_KL:-true}")
    # Drop inherited PPO options that do not apply to the GGPO objective.
    for key in actor_rollout_ref.actor.clip_ratio actor_rollout_ref.actor.clip_ratio_low \
        actor_rollout_ref.actor.clip_ratio_high actor_rollout_ref.actor.clip_ratio_c \
        actor_rollout_ref.actor.policy_loss.ppo_kl_coef actor_rollout_ref.actor.policy_loss.clip_cov_lb \
        actor_rollout_ref.actor.policy_loss.clip_cov_ub actor_rollout_ref.actor.policy_loss.clip_cov_ratio \
        actor_rollout_ref.actor.policy_loss.kl_cov_ratio actor_rollout_ref.actor.loss_agg_mode \
        algorithm.adv_estimator algorithm.gamma algorithm.lam algorithm.norm_adv_by_std_in_grpo \
        algorithm.kl_ctrl.target_kl algorithm.use_kl_in_reward algorithm.pf_ppo.reweight_method \
        algorithm.pf_ppo.weight_pow algorithm.rollout_correction.bypass_mode \
        algorithm.rollout_correction.loss_type algorithm.rollout_correction.rollout_is_batch_normalize; do
        EXTRA+=("~${key}")
    done
fi
if (( REFRESH_EVERY > 0 )); then
    EXTRA+=("trainer.ggpo_moving_reference.enabled=true"
        "trainer.ggpo_moving_reference.refresh_every_updates=${REFRESH_EVERY}")
fi
REWARD_KL_SUFFIX=
if [[ "$ALGORITHM" == ggpo && "${GGPO_REWARD_KL:-true}" == false ]]; then
    REWARD_KL_SUFFIX=-reward-only
fi
RUN_NAME=${RUN_NAME:-${MODEL_PATH##*/}-${ALGORITHM}-${MODE}-n${N}-m${GROUP_SIZE}-beta${BETA}-ref${REFRESH_EVERY}-seed${SEED}${REWARD_KL_SUFFIX}}
LOGGER='[console]'
if [[ "${WANDB_MODE:-disabled}" != disabled ]]; then LOGGER='[console,wandb]'; fi
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
OVERRIDES=(
    "algorithm.adv_estimator=grpo" "algorithm.use_kl_in_reward=false"
    "data.train_files=${DATA_ROOT}/train.parquet"
    "data.val_files=[${DATA_ROOT}/math500.parquet]"
    "data.train_batch_size=$((32 * N))"
    "data.max_prompt_length=2048" "data.max_response_length=2048"
    "data.filter_overlong_prompts=true" "data.truncation=error"
    "data.shuffle=true" "data.seed=${SEED}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.use_remove_padding=true"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=32"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-${GROUP_SIZE}}"
    "actor_rollout_ref.actor.ppo_epochs=1" "actor_rollout_ref.actor.shuffle=false"
    "actor_rollout_ref.actor.use_dynamic_bsz=false"
    "actor_rollout_ref.actor.data_loader_seed=${SEED}"
    "actor_rollout_ref.actor.entropy_coeff=0" "actor_rollout_ref.actor.calculate_entropy=true"
    "actor_rollout_ref.actor.optim.optimizer=AdamW" "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.optim.betas=[0.9,0.999]" "actor_rollout_ref.actor.optim.weight_decay=0.01"
    "actor_rollout_ref.actor.optim.lr_scheduler_type=constant"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=0" "actor_rollout_ref.actor.optim.clip_grad=1.0"
    "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32"
    "actor_rollout_ref.actor.fsdp_config.dtype=bfloat16"
    "actor_rollout_ref.actor.fsdp_config.seed=${SEED}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-true}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-true}"
    "actor_rollout_ref.ref.fsdp_config.model_dtype=fp32"
    "actor_rollout_ref.ref.fsdp_config.dtype=bfloat16"
    "actor_rollout_ref.ref.fsdp_config.seed=${SEED}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=true"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384"
    "actor_rollout_ref.rollout.name=vllm" "actor_rollout_ref.rollout.dtype=bfloat16"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
    "actor_rollout_ref.rollout.max_model_len=4096"
    "actor_rollout_ref.rollout.max_num_batched_tokens=8192"
    "actor_rollout_ref.rollout.max_num_seqs=256"
    "actor_rollout_ref.rollout.skip_tokenizer_init=${ROLLOUT_SKIP_TOKENIZER_INIT:-False}"
    "actor_rollout_ref.rollout.n=${GROUP_SIZE}" "actor_rollout_ref.rollout.seed=${SEED}"
    "actor_rollout_ref.rollout.temperature=1.0" "actor_rollout_ref.rollout.top_p=1.0"
    "actor_rollout_ref.rollout.top_k=-1" "+actor_rollout_ref.rollout.repetition_penalty=1.0"
    "actor_rollout_ref.rollout.calculate_log_probs=true"
    "actor_rollout_ref.rollout.logprobs_mode=processed_logprobs"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.val_kwargs.n=32" "actor_rollout_ref.rollout.val_kwargs.do_sample=true"
    "actor_rollout_ref.rollout.val_kwargs.temperature=1.0"
    "actor_rollout_ref.rollout.val_kwargs.top_p=1.0" "actor_rollout_ref.rollout.val_kwargs.top_k=-1"
    "+algorithm.filter_groups.enable=true" "+algorithm.filter_groups.metric=raw_correctness"
    "+algorithm.filter_groups.max_num_gen_batches=32"
    "reward.reward_manager.name=naive"
    "reward.custom_reward_function.path=${SCRIPT_DIR}/math_verifier.py"
    "reward.custom_reward_function.name=compute_score"
    "critic.enable=false" "trainer.balance_batch=false" "trainer.use_v1=false"
    "trainer.consume_collection_minibatches=true"
    "trainer.total_training_steps=${TOTAL_UPDATES}" "trainer.total_epochs=1000"
    "trainer.save_freq=${SAVE_EVERY}" "trainer.test_freq=${EVAL_EVERY}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE:-4}" "trainer.nnodes=${NNODES:-1}"
    "trainer.default_local_dir=${CKPTS_DIR:-checkpoints/${RUN_NAME}}"
    "trainer.project_name=${WANDB_PROJECT:-GGPO}" "trainer.experiment_name=${RUN_NAME}"
    "trainer.logger=${LOGGER}"
    "${EXTRA[@]}" "$@"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'OVERRIDE %s\n' "${OVERRIDES[@]}"
    exit 0
fi
cd "$REPO_ROOT"
exec python3 -m verl.trainer.main_ppo "${OVERRIDES[@]}"
