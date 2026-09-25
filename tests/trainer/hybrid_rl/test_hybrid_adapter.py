import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.hybrid_rl.trainer import (
    BOUNDARY_METRIC_KEYS,
    HISTORICAL,
    ON_POLICY,
    HybridCursorLoader,
    HybridGGPOTrainer,
    HybridRolloutAdapter,
    assert_hybrid_plan,
    build_hybrid_plan,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _stored_batch(plan, *, ordered=True, source_indices=None, group_prompts=None):
    count = 256 * 8
    response_order = torch.from_numpy(plan.response_order) if ordered else torch.arange(count)
    response_owner = torch.arange(256).repeat_interleave(8)[response_order]
    response_slot = torch.arange(8).repeat(256)[response_order]
    if group_prompts is None:
        group_prompts = torch.arange(256)[:, None].repeat(1, 2)
    prompts = group_prompts[response_owner]
    if source_indices is None:
        source_indices = np.arange(256)
    ordered_source_indices = np.asarray(source_indices)
    if ordered:
        ordered_source_indices = ordered_source_indices[np.asarray(plan.group_order)]
    responses = response_slot[:, None].repeat(1, 3)
    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    response_mask = torch.ones_like(responses, dtype=torch.bool)
    scores = torch.zeros_like(responses, dtype=torch.float32)
    scores[:, -1] = (response_owner % 2).float()
    return DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": torch.arange(5).repeat(count, 1),
                "response_mask": response_mask,
                "rollout_log_probs": torch.full((count, 3), -2.0),
                "token_level_scores": scores,
            },
            batch_size=count,
        ),
        non_tensor_batch={
            "source_index": np.repeat(ordered_source_indices, 8),
        },
    )


def test_plan_is_deterministic_group_contiguous_and_stratified():
    first = build_hybrid_plan(37, 42)
    second = build_hybrid_plan(37, 42)
    assert first == second
    assert_hybrid_plan(first)
    assert len(set(first.group_order)) == 256
    sources = first.ordered_group_sources
    assert np.count_nonzero(sources == HISTORICAL) == 192
    assert np.count_nonzero(sources == ON_POLICY) == 64
    for index in range(8):
        chunk = sources[index * 32 : (index + 1) * 32]
        assert np.count_nonzero(chunk == HISTORICAL) == 24
        assert np.count_nonzero(chunk == ON_POLICY) == 8
    response_order = first.response_order.reshape(256, 8)
    assert np.all(np.diff(response_order, axis=1) == 1)


class _HistoricalDataset:
    manifest = {
        "identity": {
            "source": {"path": "/test/train.parquet"},
            "behavior_model": {"id": "model", "revision": "model-revision"},
            "geometry": {"max_prompt_length": 2},
        }
    }

    def load_batch(self, index):
        assert index == 0
        return _stored_batch(build_hybrid_plan(1), ordered=False)


class _Tokenizer:
    name_or_path = "test-tokenizer"
    init_kwargs = {"_commit_hash": "test-tokenizer-revision"}
    chat_template = "test-template"

    def apply_chat_template(self, raw_prompt, *, add_generation_prompt, tokenize, **kwargs):
        assert add_generation_prompt is True
        assert tokenize is True
        return raw_prompt


class _TrainDataset:
    processor = None
    tokenizer = _Tokenizer()
    apply_chat_template_kwargs = {}
    tool_schemas = None

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _row(source_dataset, index, prompt):
    return {
        "dummy_tensor": torch.tensor([index]),
        "raw_prompt": list(prompt),
        "index": index,
        "extra_info": {
            "index": index,
            "source_dataset": source_dataset,
            "source_revision": f"{source_dataset}-revision",
        },
    }


def _collate(rows):
    return {
        "dummy_tensor": torch.stack([row["dummy_tensor"] for row in rows]),
        "raw_prompt": np.asarray([row["raw_prompt"] for row in rows], dtype=object),
        "source_identity": np.asarray(
            [(row["extra_info"]["source_dataset"], row["index"]) for row in rows], dtype=object
        ),
    }


def _loader(train, historical=None):
    return HybridCursorLoader(
        train,
        _collate,
        _HistoricalDataset() if historical is None else historical,
        42,
        100,
    )


def test_cursor_state_reproduces_plan_and_next_shard():
    train = _TrainDataset(list(reversed([_row("source", index, [index, index]) for index in range(256)])))
    loader = _loader(train)
    batch = next(iter(loader))
    assert batch["_hybrid_source"].shape == (256,)
    state = loader.state_dict()
    assert state["completed_optimizer_steps"] == 1
    assert state["next_shard_index"] == 1
    assert state["schema_version"] == 2
    assert state["prompt_identity_schema"] == "source-dataset-index+stored-prompt-tokens-v1"
    assert loader.source_index_to_positions[0] == [255]
    assert len(state["last_completed_plan"]["fresh_group_mask"]) == 256
    assert len(state["last_completed_plan"]["minibatch_seeds"]) == 8

    resumed = _loader(train)
    resumed.load_state_dict(state)
    assert resumed.cursor == 1
    assert resumed.last_completed_plan == build_hybrid_plan(1)


class _IdentityCollisionHistoricalDataset(_HistoricalDataset):
    def load_batch(self, index):
        assert index == 0
        source_indices = np.arange(256)
        source_indices[1] = 0
        return _stored_batch(build_hybrid_plan(1), ordered=False, source_indices=source_indices)


def test_same_bare_index_from_two_sources_resolves_by_stored_prompt_content():
    rows = [_row("guru", index, [index, index]) for index in range(256)]
    rows[1] = _row("openr1", 0, [1, 1])
    loader = _loader(_TrainDataset(rows), _IdentityCollisionHistoricalDataset())

    batch = next(iter(loader))

    identities = [tuple(value) for value in batch["source_identity"]]
    assert ("guru", 0) in identities
    assert ("openr1", 0) in identities
    assert len(set(identities)) == 256


class _DuplicatePromptHistoricalDataset(_HistoricalDataset):
    def load_batch(self, index):
        assert index == 0
        group_prompts = torch.arange(256)[:, None].repeat(1, 2)
        group_prompts[1] = group_prompts[0]
        return _stored_batch(build_hybrid_plan(1), ordered=False, group_prompts=group_prompts)


def test_true_duplicate_prompt_content_is_rejected():
    rows = [_row("source", index, [index, index]) for index in range(256)]
    rows[1]["raw_prompt"] = [0, 0]
    loader = _loader(_TrainDataset(rows), _DuplicatePromptHistoricalDataset())

    with np.testing.assert_raises_regex(ValueError, "duplicate prompt content"):
        next(iter(loader))


class _AmbiguousPromptHistoricalDataset(_HistoricalDataset):
    def load_batch(self, index):
        assert index == 0
        source_indices = np.arange(256)
        source_indices[0] = 999
        return _stored_batch(build_hybrid_plan(1), ordered=False, source_indices=source_indices)


def test_multiple_composite_candidates_with_same_prompt_fail_closed():
    rows = [_row("source", index, [index, index]) for index in range(1, 256)]
    rows.extend([_row("guru", 999, [0, 0]), _row("openr1", 999, [0, 0])])
    loader = _loader(_TrainDataset(rows), _AmbiguousPromptHistoricalDataset())

    with np.testing.assert_raises_regex(ValueError, "resolved to 2 ALP rows"):
        next(iter(loader))


class _MissingPromptHistoricalDataset(_HistoricalDataset):
    def load_batch(self, index):
        assert index == 0
        source_indices = np.arange(256)
        source_indices[0] = 999
        return _stored_batch(build_hybrid_plan(1), ordered=False, source_indices=source_indices)


def test_missing_composite_candidate_fails_closed():
    rows = [_row("source", index, [index, index]) for index in range(256)]
    loader = _loader(_TrainDataset(rows), _MissingPromptHistoricalDataset())

    with pytest.raises(ValueError, match="resolved to 0 ALP rows"):
        next(iter(loader))


class _SplitGroupHistoricalDataset(_HistoricalDataset):
    def load_batch(self, index):
        assert index == 0
        batch = _stored_batch(build_hybrid_plan(1), ordered=False)
        batch.non_tensor_batch["source_index"][1] = 999
        return batch


def test_group_with_inconsistent_response_identity_is_rejected():
    rows = [_row("source", index, [index, index]) for index in range(256)]
    loader = _loader(_TrainDataset(rows), _SplitGroupHistoricalDataset())

    with pytest.raises(ValueError, match="one source index across 8 responses"):
        next(iter(loader))


class _LeftPaddedHistoricalDataset(_HistoricalDataset):
    def load_batch(self, index):
        assert index == 0
        group_prompts = torch.stack([torch.full((256,), 999), torch.arange(256)], dim=1)
        batch = _stored_batch(build_hybrid_plan(1), ordered=False, group_prompts=group_prompts)
        batch.batch["attention_mask"][:, 0] = 0
        return batch


def test_stored_prompt_identity_ignores_only_left_padding():
    rows = [_row("source", index, [index]) for index in range(256)]
    loader = _loader(_TrainDataset(rows), _LeftPaddedHistoricalDataset())

    batch = next(iter(loader))

    assert len(batch["source_identity"]) == 256


class _FreshManager:
    def __init__(self, *, omit_rm_scores=False):
        self.received = None
        self.omit_rm_scores = omit_rm_scores

    def generate_sequences(self, prompts):
        self.received = prompts
        count = len(prompts)
        tensors = TensorDict(
            {
                "prompts": torch.full((count, 2), 9),
                "responses": torch.full((count, 3), 9),
                "input_ids": torch.full((count, 5), 9),
                "attention_mask": torch.ones((count, 5), dtype=torch.long),
                "position_ids": torch.arange(5).repeat(count, 1),
                "response_mask": torch.ones((count, 3), dtype=torch.long),
                "rollout_log_probs": torch.full((count, 3), -9.0),
                "rm_scores": torch.nn.functional.pad(torch.ones((count, 1)), (2, 0)),
            },
            batch_size=count,
        )
        if self.omit_rm_scores:
            del tensors["rm_scores"]
        return DataProto(
            batch=tensors,
            non_tensor_batch={
                "__num_turns__": np.ones(count, dtype=np.int32),
                "acc": np.ones(count, dtype=np.float32),
                "data_source": prompts.non_tensor_batch["data_source"].copy(),
                "valid_response_length": np.full(count, 3, dtype=np.int32),
                "multi_modal_inputs": np.asarray([{} for _ in range(count)], dtype=object),
            },
            meta_info={"timing": {"generate_sequences": 1.0}, "reward_extra_keys": ["acc", "valid_response_length"]},
        )


def test_rollout_adapter_passes_validation_through_without_historical_shard():
    prompts = DataProto(
        batch=TensorDict({"dummy_tensor": torch.zeros((3, 1))}, batch_size=3),
        non_tensor_batch={"data_source": np.full(3, "alp_math", dtype=object)},
        meta_info={"validate": True},
    )
    loader = type("Loader", (), {"active_plan": None, "active_historical_batch": None})()
    manager = _FreshManager()

    output = HybridRolloutAdapter(manager, loader).generate_sequences(prompts)

    assert manager.received is prompts
    assert len(output) == 3


def test_rollout_adapter_generates_only_fresh_groups_and_stitches_full_batch():
    plan = build_hybrid_plan(1)
    stored = _stored_batch(plan)
    sources = np.repeat(plan.ordered_group_sources, 8)
    prompts = DataProto(
        batch=TensorDict({"dummy_tensor": torch.zeros((2048, 1))}, batch_size=2048),
        non_tensor_batch={
            "_hybrid_source": sources,
            "_hybrid_shard_index": np.zeros(2048, dtype=np.int32),
            "_hybrid_group_index": np.repeat(np.asarray(plan.group_order), 8),
            "data_source": np.full(2048, "alp_math", dtype=object),
        },
    )
    loader = type("Loader", (), {"active_plan": plan, "active_historical_batch": stored})()
    manager = _FreshManager()
    output = HybridRolloutAdapter(manager, loader).generate_sequences(prompts)

    assert len(manager.received) == 64 * 8
    assert len(output) == 256 * 8
    fresh = sources == ON_POLICY
    historical = ~fresh
    assert torch.all(output.batch["responses"][torch.from_numpy(fresh)] == 9)
    torch.testing.assert_close(
        output.batch["responses"][torch.from_numpy(historical)],
        stored.batch["responses"][torch.from_numpy(historical)],
    )
    assert np.array_equal(output.non_tensor_batch["_hybrid_source"], sources)
    assert np.array_equal(output.non_tensor_batch["data_source"], prompts.non_tensor_batch["data_source"])
    assert output.meta_info["hybrid_composition"] == "192H+64O"


def test_rollout_adapter_rejects_fresh_responses_without_rewards():
    plan = build_hybrid_plan(1)
    sources = np.repeat(plan.ordered_group_sources, 8)
    prompts = DataProto(
        batch=TensorDict({"dummy_tensor": torch.zeros((2048, 1))}, batch_size=2048),
        non_tensor_batch={
            "_hybrid_source": sources,
            "data_source": np.full(2048, "alp_math", dtype=object),
        },
    )
    loader = type(
        "Loader",
        (),
        {"active_plan": plan, "active_historical_batch": _stored_batch(plan)},
    )()

    with pytest.raises(ValueError, match="must return rm_scores"):
        HybridRolloutAdapter(_FreshManager(omit_rm_scores=True), loader).generate_sequences(prompts)


def test_hybrid_checkpoint_state_is_durable_before_parent_publishes_boundary(tmp_path, monkeypatch):
    trainer = HybridGGPOTrainer.__new__(HybridGGPOTrainer)
    trainer.global_steps = 20
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(tmp_path), experiment_name="hybrid-test"),
        hybrid_data=SimpleNamespace(dataset_id="historical-test"),
    )
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"completed_optimizer_steps": 20})
    trainer._checkpoint_boundary_metrics = {key: 0.5 for key in BOUNDARY_METRIC_KEYS}
    trainer._checkpoint_boundary_metrics["training/global_step"] = 20
    trainer._checkpoint_boundary_metrics["hybrid/historical_groups"] = 192
    trainer._checkpoint_boundary_metrics["hybrid/on_policy_groups"] = 64
    monkeypatch.setenv("WANDB_RUN_ID", "immutable-test-run")

    observed = {}

    def parent_save(self):
        path = tmp_path / "global_step_20" / "hybrid_state.json"
        observed["state"] = json.loads(path.read_text())
        observed["metrics"] = json.loads(path.with_name("boundary_metrics.json").read_text())

    monkeypatch.setattr(RayPPOTrainer, "_save_checkpoint", parent_save)

    trainer._save_checkpoint()

    assert observed["state"]["wandb_run_id"] == "immutable-test-run"
    assert observed["state"]["completed_optimizer_steps"] == 20
    assert observed["metrics"]["metrics"]["training/global_step"] == 20


def test_hybrid_auto_resume_uses_parent_resolved_checkpoint_path(tmp_path, monkeypatch):
    folder = tmp_path / "global_step_20"
    folder.mkdir()
    (folder / "hybrid_state.json").write_text(
        json.dumps(
            {
                "completed_optimizer_steps": 20,
                "next_shard_index": 20,
                "dataset_id": "historical-test",
                "experiment_name": "hybrid-test",
                "wandb_run_id": "immutable-test-run",
                "objective": "ggpo-exp-seqmean-beta0.001",
            }
        )
    )
    trainer = HybridGGPOTrainer.__new__(HybridGGPOTrainer)
    trainer.global_steps = 0
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(resume_mode="auto", resume_from_path=None, experiment_name="hybrid-test"),
        hybrid_data=SimpleNamespace(
            dataset_id="historical-test", segment_optimizer_steps=20, target_optimizer_steps=100
        ),
    )
    trainer.train_dataloader = SimpleNamespace(cursor=20)
    monkeypatch.setenv("WANDB_RUN_ID", "immutable-test-run")

    def parent_load(self):
        self.global_steps = 20
        return str(folder)

    monkeypatch.setattr(RayPPOTrainer, "_load_checkpoint", parent_load)

    trainer._load_checkpoint()

    assert trainer.global_steps == 20
    assert trainer.total_training_steps == 40


def test_hybrid_completed_checkpoint_allows_validation_only(tmp_path, monkeypatch):
    folder = tmp_path / "global_step_100"
    folder.mkdir()
    (folder / "hybrid_state.json").write_text(
        json.dumps(
            {
                "completed_optimizer_steps": 100,
                "next_shard_index": 100,
                "dataset_id": "historical-test",
                "experiment_name": "hybrid-test",
                "wandb_run_id": "immutable-test-run",
                "objective": "ggpo-exp-seqmean-beta0.001",
            }
        )
    )
    trainer = HybridGGPOTrainer.__new__(HybridGGPOTrainer)
    trainer.global_steps = 0
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            resume_mode="resume_path",
            resume_from_path=str(folder),
            experiment_name="hybrid-test",
            val_only=True,
        ),
        hybrid_data=SimpleNamespace(
            dataset_id="historical-test", segment_optimizer_steps=20, target_optimizer_steps=100
        ),
    )
    trainer.train_dataloader = SimpleNamespace(cursor=100)
    monkeypatch.setenv("WANDB_RUN_ID", "immutable-test-run")

    def parent_load(self):
        self.global_steps = 100
        return str(folder)

    monkeypatch.setattr(RayPPOTrainer, "_load_checkpoint", parent_load)

    trainer._load_checkpoint()

    assert trainer.global_steps == 100
    assert trainer.total_training_steps == 100


def test_hybrid_validation_uses_only_canonical_visible_namespace(monkeypatch):
    trainer = HybridGGPOTrainer.__new__(HybridGGPOTrainer)
    trainer.global_steps = 20
    monkeypatch.setattr(
        RayPPOTrainer,
        "_validate",
        lambda self, merged=False: {"val-core/alp_math500/acc/mean@32": 0.5},
    )

    metrics = trainer._validate()

    assert metrics["training/global_step"] == 20
    assert all(not key.startswith("validation/") for key in metrics)
