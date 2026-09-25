from pathlib import Path

import pytest
from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler

from verl.experimental.fully_async_policy.fully_async_main import (
    describe_fully_async_mode,
    effective_rollout_contract,
    validate_fully_async_config,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _config(
    *,
    backend="verl",
    loss_mode="vanilla",
    max_lag=None,
    complete=True,
    hybrid=False,
    resource_hybrid_engine=False,
    use_kl_in_reward=False,
    val_before_train=False,
    test_freq=-1,
    actor_strategy="fsdp2",
    max_optimizer_steps=None,
    controlled_stop_after_seconds=None,
    controlled_stop_deadline_unix_seconds=None,
    adaptive_controlled_stop=None,
    resume_mode="auto",
    trigger_parameter_sync_step=2,
):
    algorithm = {}
    if loss_mode != "ggpo":
        algorithm.update(
            {
                "adv_estimator": "grpo",
                "norm_adv_by_std_in_grpo": loss_mode != "asympo",
                "use_kl_in_reward": use_kl_in_reward,
            }
        )
    elif use_kl_in_reward:
        # Deliberately inject this inert GGPO key for the rejection test.
        algorithm["use_kl_in_reward"] = True

    return OmegaConf.create(
        {
            "async_training": {
                "rollout_backend": backend,
                "max_policy_version_lag": max_lag,
                "require_complete_groups": complete,
                "use_trainer_do_validate": hybrid,
                "runtime_root": None,
                "protected_roots": [],
                "trigger_parameter_sync_step": trigger_parameter_sync_step,
                "areal": {
                    "enable_thinking": True,
                    "max_head_offpolicyness": max_lag,
                },
            },
            "rollout": {"nnodes": 1, "n_gpus_per_node": 4, "n": 2},
            "actor_rollout_ref": {
                "hybrid_engine": resource_hybrid_engine,
                "actor": {
                    "strategy": actor_strategy,
                    "policy_loss": {
                        "loss_mode": loss_mode,
                        "beta": 0.001 if loss_mode in {"ggpo", "agro"} else 0.1,
                        "sequence_log_prob_reduction": "sum" if loss_mode == "agro" else "mean",
                    },
                },
                "rollout": {
                    "nnodes": 1,
                    "n_gpus_per_node": 4,
                    "n": 2,
                    "seed": 42,
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.6,
                    "max_num_seqs": 256,
                    "max_num_batched_tokens": 8192,
                    "enforce_eager": True,
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "engine_kwargs": {"vllm": {}, "sglang": {}},
                },
            },
            "data": {
                "train_batch_size": 0,
                "apply_chat_template_kwargs": {"enable_thinking": True},
                "chat_template_func": None,
                "filter_overlong_prompts": True,
                "truncation": "error",
            },
            "algorithm": algorithm,
            "reward": {
                "reward_manager": {"source": "register", "name": "batch"},
                "custom_reward_function": {"path": None},
            },
            "trainer": {
                "val_before_train": val_before_train,
                "test_freq": test_freq,
                "max_optimizer_steps": max_optimizer_steps,
                "controlled_stop_after_seconds": controlled_stop_after_seconds,
                "controlled_stop_deadline_unix_seconds": controlled_stop_deadline_unix_seconds,
                "adaptive_controlled_stop": adaptive_controlled_stop,
                "resume_mode": resume_mode,
            },
        }
    )


def test_verl_grpo_preserves_legacy_optional_version_cap():
    config = _config()
    validate_fully_async_config(config)
    assert describe_fully_async_mode(config) == "algorithm=grpo backend=verl advantage_estimator=grpo"


def test_ggpo_mode_is_explicit_in_smoke_gate_log():
    config = _config(loss_mode="ggpo", max_lag=0)
    validate_fully_async_config(config)
    assert describe_fully_async_mode(config) == (
        "algorithm=ggpo backend=verl beta=0.001 reduction=mean group_size=2"
    )


def test_ggpo_accepts_configured_beta_and_reduction():
    config = _config(loss_mode="ggpo", max_lag=0)
    policy_loss = config.actor_rollout_ref.actor.policy_loss
    policy_loss.beta = 0.25
    policy_loss.sequence_log_prob_reduction = "sum"
    validate_fully_async_config(config)

    assert describe_fully_async_mode(config) == (
        "algorithm=ggpo backend=verl beta=0.25 reduction=sum group_size=2"
    )


def test_agro_and_asympo_modes_remain_dedicated_in_smoke_gate_log():
    agro = _config(loss_mode="agro", max_lag=0)
    agro.actor_rollout_ref.actor.policy_loss.beta = 0.25
    asympo = _config(loss_mode="asympo", max_lag=0)
    validate_fully_async_config(agro)
    validate_fully_async_config(asympo)
    assert describe_fully_async_mode(agro) == (
        "algorithm=agro backend=verl beta=0.25 reduction=sum group_size=2"
    )
    assert describe_fully_async_mode(asympo) == "algorithm=asympo backend=verl reduction=response-equal"


def test_areal_backend_accepts_native_validation_without_hybrid_replicas():
    config = _config(backend="areal", max_lag=0, val_before_train=True, test_freq=1)
    validate_fully_async_config(config)
    assert describe_fully_async_mode(config) == "algorithm=grpo backend=areal advantage_estimator=grpo"


def test_areal_registered_reward_must_use_batch_manager_before_ray_starts():
    config = OmegaConf.merge(
        _config(backend="areal", max_lag=0),
        {"reward": {"reward_manager": {"name": "naive"}}},
    )
    with pytest.raises(ValueError, match="reward.reward_manager.name=batch"):
        validate_fully_async_config(config)


def test_effective_rollout_contract_is_backend_independent():
    verl_config = _config(backend="verl", max_lag=4)
    areal_config = _config(backend="areal", max_lag=4)
    validate_fully_async_config(verl_config)
    validate_fully_async_config(areal_config)

    verl_contract = effective_rollout_contract(verl_config)
    areal_contract = effective_rollout_contract(areal_config)
    assert verl_contract | {"backend": "areal"} == areal_contract
    assert areal_contract["seed"] == 42
    assert areal_contract["sampling"] == {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
    }
    assert areal_contract["resources"] == {
        "nnodes": 1,
        "n_gpus_per_node": 4,
        "responses_per_prompt": 2,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }


def test_conflicting_public_rollout_duplicate_fails_closed():
    config = _config(max_lag=4)
    config.rollout.n = 8
    with pytest.raises(ValueError, match="Conflicting duplicate rollout settings"):
        validate_fully_async_config(config)


def test_conflicting_native_engine_override_fails_closed():
    config = _config(max_lag=4)
    config.actor_rollout_ref.rollout.engine_kwargs.vllm.seed = 7
    with pytest.raises(ValueError, match="Conflicting duplicate rollout settings"):
        validate_fully_async_config(config)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"async_training": {"areal": {"enable_thinking": False}}}, "enable_thinking"),
        ({"data": {"filter_overlong_prompts": False}}, "filter_overlong_prompts"),
        ({"data": {"truncation": "left"}}, "truncation=error"),
        ({"data": {"chat_template_func": "custom.fn"}}, "chat_template_func"),
        (
            {"async_training": {"areal": {"max_head_offpolicyness": 3}}},
            "max_head_offpolicyness",
        ),
    ],
)
def test_areal_preprocessing_and_admission_drift_fail_closed(override, message):
    config = OmegaConf.merge(_config(backend="areal", max_lag=4), override)
    with pytest.raises(ValueError, match=message):
        validate_fully_async_config(config)


def test_declared_smoke_runtime_is_no_checkpoint_and_contained(tmp_path):
    root = tmp_path / "smoke"
    root.mkdir()
    config = OmegaConf.merge(
        _config(backend="areal", max_lag=4),
        {
            "async_training": {
                "runtime_root": str(root),
                "weight_sync": {"export_dir": str(root / "cache" / "weights")},
                "areal": {
                    "fileroot": str(root / "cache" / "areal"),
                    "name_resolve_root": str(root / "cache" / "name_resolve"),
                },
            },
            "trainer": {
                "save_freq": -1,
                "save_final_checkpoint": False,
                "default_local_dir": str(root / "artifacts"),
            },
        },
    )
    validate_fully_async_config(config)

    config.async_training.areal.fileroot = str(tmp_path / "escape")
    with pytest.raises(ValueError, match="escapes smoke runtime root"):
        validate_fully_async_config(config)




def test_disabled_validation_does_not_require_or_construct_a_validation_dataset():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {
                "train_files": ["unused"],
                "val_files": [],
                "train_batch_size": 0,
                "gen_batch_size": 1,
                "val_batch_size": None,
                "dataloader_num_workers": 0,
            },
            "trainer": {
                "val_before_train": False,
                "test_freq": -1,
                "total_epochs": 1,
                "total_training_steps": None,
            },
        }
    )
    train_dataset = [{"prompt": "one"}, {"prompt": "two"}]

    trainer._create_dataloader(
        train_dataset=train_dataset,
        val_dataset=None,
        collate_fn=lambda rows: rows,
        train_sampler=SequentialSampler(train_dataset),
    )

    assert trainer.val_dataset is None
    assert trainer.val_dataloader is None
    assert len(trainer.train_dataloader) == 2


def test_packaged_async_config_declares_areal_runtime_paths_for_hydra_overrides():
    config_path = (
        Path(__file__).resolve().parents[3]
        / "verl"
        / "experimental"
        / "fully_async_policy"
        / "config"
        / "fully_async_ppo_trainer.yaml"
    )
    config = OmegaConf.load(config_path)

    assert config.async_training.areal.backend is None
    assert config.async_training.areal.fileroot == "./areal_runs"
    assert config.async_training.areal.name_resolve_root == "./areal_runs/name_resolve"
    assert config.async_training.areal.validation_max_concurrent_rollouts == 8
    assert config.trainer.save_final_checkpoint is True
    assert config.trainer.max_optimizer_steps is None
    assert config.trainer.controlled_stop_after_seconds is None
    assert config.trainer.controlled_stop_deadline_unix_seconds is None
    assert config.trainer.adaptive_controlled_stop is None


@pytest.mark.parametrize("resume_mode", ["disable", "auto", "resume_path"])
def test_bounded_optimizer_run_is_a_per_process_cap(resume_mode):
    """A resumed process must be able to execute its own exact update budget."""
    validate_fully_async_config(_config(max_optimizer_steps=20, resume_mode=resume_mode))


@pytest.mark.parametrize("resume_mode", ["disable", "auto", "resume_path"])
def test_controlled_wall_clock_stop_is_process_scoped(resume_mode):
    validate_fully_async_config(_config(controlled_stop_after_seconds=6300, resume_mode=resume_mode))


@pytest.mark.parametrize("resume_mode", ["disable", "auto", "resume_path"])
def test_absolute_controlled_wall_clock_stop_is_process_scoped(resume_mode):
    validate_fully_async_config(
        _config(controlled_stop_deadline_unix_seconds=1_800_000_000, resume_mode=resume_mode)
    )


@pytest.mark.parametrize("sync_steps", [1, 2, 4])
def test_adaptive_controlled_stop_accepts_dynamic_sync_window(sync_steps):
    validate_fully_async_config(
        _config(
            adaptive_controlled_stop={
                "hard_deadline_unix_seconds": 1_800_000_000,
                "initial_optimizer_step_seconds": 300,
                "finalization_seconds": 60,
                "safety_factor": 1.25,
                "history_size": 8,
            },
            trigger_parameter_sync_step=sync_steps,
        )
    )


def test_controlled_wall_clock_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_fully_async_config(
            _config(
                controlled_stop_after_seconds=6300,
                controlled_stop_deadline_unix_seconds=1_800_000_000,
                resume_mode="disable",
            )
        )


def test_adaptive_and_legacy_controlled_wall_clock_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_fully_async_config(
            _config(
                controlled_stop_deadline_unix_seconds=1_800_000_000,
                adaptive_controlled_stop={
                    "hard_deadline_unix_seconds": 1_800_000_000,
                    "initial_optimizer_step_seconds": 300,
                    "finalization_seconds": 60,
                },
            )
        )


def test_adaptive_controlled_stop_rejects_partial_terminal_sync_window():
    with pytest.raises(ValueError, match="must end on a complete"):
        validate_fully_async_config(
            _config(
                max_optimizer_steps=10,
                adaptive_controlled_stop={
                    "hard_deadline_unix_seconds": 1_800_000_000,
                    "initial_optimizer_step_seconds": 300,
                    "finalization_seconds": 60,
                },
                trigger_parameter_sync_step=4,
            )
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_config(backend="unknown"), "rollout_backend"),
        (_config(complete=False), "require_complete_groups"),
        (_config(resource_hybrid_engine=True), "hybrid_engine=false"),
        (OmegaConf.merge(_config(), {"data": {"train_batch_size": 64}}), "train_batch_size=0"),
        (_config(max_lag=-1), "non-negative"),
        (_config(max_lag=True), "non-negative integer"),
        (_config(max_optimizer_steps=0), "max_optimizer_steps"),
        (_config(max_optimizer_steps=-1), "max_optimizer_steps"),
        (_config(max_optimizer_steps=True), "max_optimizer_steps"),
        (_config(max_optimizer_steps=1.5), "max_optimizer_steps"),
        (_config(controlled_stop_after_seconds=0), "controlled_stop_after_seconds"),
        (_config(controlled_stop_after_seconds=-1), "controlled_stop_after_seconds"),
        (_config(controlled_stop_after_seconds=True), "controlled_stop_after_seconds"),
        (
            _config(controlled_stop_deadline_unix_seconds=0),
            "controlled_stop_deadline_unix_seconds",
        ),
        (
            _config(controlled_stop_deadline_unix_seconds=True),
            "controlled_stop_deadline_unix_seconds",
        ),
        (_config(backend="areal"), "explicit"),
        (_config(backend="areal", max_lag=0, hybrid=True), "hybrid validation"),
        (_config(backend="areal", max_lag=0, actor_strategy="megatron"), "actor.strategy=fsdp"),
        (_config(loss_mode="ggpo"), "explicit"),
        (
            _config(loss_mode="ggpo", max_lag=0, use_kl_in_reward=True),
            "use_kl_in_reward=false",
        ),
    ],
)
def test_invalid_async_backend_combinations_fail_before_ray_starts(config, message):
    with pytest.raises(ValueError, match=message):
        validate_fully_async_config(config)
