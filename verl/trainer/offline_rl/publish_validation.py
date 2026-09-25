# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Publish one checkpoint-bound Math500 result to its parent W&B lineage."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALIDATION_METRICS = (
    "val-core/alp_math500/acc/mean@32",
    "val-core/alp_math500/acc/best@32/mean",
    "val-core/alp_math500/acc/best@32/std",
)
STEP_METRIC = "training/global_step"
AUDIT_PREFIX = "_offline_validation"


def validation_internal_step(optimizer_step: int) -> int:
    """Reserve one W&B History row after every 20-step training boundary.

    The 50-round cut publishes at 50 as well. Three validation rows (0/20/40)
    precede it, so its reserved row is 53; step // 20 would say 52, which is
    round 50's own training row. Rounding up leaves every 20-boundary alone.
    """
    step = int(optimizer_step)
    if step < 0 or step > 100 or (step % 20 != 0 and step != 50):
        raise ValueError("validation step must be one of 0,20,40,50,60,80,100")
    return step + -(-step // 20)


def training_history_offset(resume_step: int) -> int:
    step = int(resume_step)
    if step < 0 or step >= 100 or step % 20 != 0:
        raise ValueError("training resume step must be one of 0,20,40,60,80")
    return step // 20


def _finite_float(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{description} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} is not finite")
    return result


def load_evaluation_receipt(
    path: Path, *, expected_run_id: str, expected_name: str
) -> tuple[dict[str, Any], int, dict[str, float]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("evaluation receipt must be one immutable regular file")
    document = json.loads(path.read_text(encoding="utf-8"))
    step = int(document.get("optimizer_step", -1))
    if (
        document.get("schema_version") != 2
        or document.get("status") != "accepted"
        or int(document.get("user_step", -1)) != step
        or document.get("wandb_run_id") != expected_run_id
        or document.get("experiment_name") != expected_name
        or int(document.get("wandb_step_offset", -1)) != step // 20
    ):
        raise ValueError("evaluation receipt identity is inconsistent")
    validation_internal_step(step)
    checkpoint = Path(str(document.get("checkpoint", ""))).resolve()
    if checkpoint.name != f"global_step_{step}" or not checkpoint.is_dir():
        raise ValueError("evaluation receipt checkpoint is missing or has the wrong boundary")
    raw_metrics = document.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise ValueError("evaluation receipt has no metric map")
    metrics = {key: _finite_float(raw_metrics.get(key), key) for key in VALIDATION_METRICS}
    if any(value < 0.0 or value > 1.0 for value in metrics.values()):
        raise ValueError("Math500 validation metric lies outside [0,1]")
    return document, step, metrics


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"publication receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _not_found(error: Exception) -> bool:
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) == 404:
        return True
    message = str(error).lower()
    return "could not find run" in message or "404" in message or "not found" in message


def _remote_identity(run: Any) -> tuple[str, str, str]:
    # id/entity/project only: the display name is regenerated from the resolved config by
    # every training segment, so it is not part of the lineage identity.
    return str(run.id), str(run.entity), str(run.project)


def _summary_index(summary: Any) -> dict[str, dict[str, Any]]:
    """Return the audit index as ``{optimizer_step: {field: value}}``.

    The writer stores a nested dict under ``_offline_validation_index``. The live SDK Summary
    keeps it nested, but the public API reads it back flattened into dotted keys such as
    ``_offline_validation_index.20.internal_step``; accept both shapes.
    """
    prefix = f"{AUDIT_PREFIX}_index"
    index: dict[str, dict[str, Any]] = {}
    nested = summary.get(prefix)
    if callable(getattr(nested, "keys", None)):
        for step in nested.keys():
            entry = nested[step]
            if callable(getattr(entry, "keys", None)):
                index[str(step)] = {str(field): entry[field] for field in entry.keys()}
    for key in summary.keys():
        if isinstance(key, str) and key.startswith(prefix + "."):
            step, _, field = key[len(prefix) + 1 :].partition(".")
            if field:
                index.setdefault(step, {})[field] = summary[key]
    return index


def _read_predecessor(
    wandb: Any,
    *,
    locator: str,
    identity: tuple[str, str, str],
    internal_step: int,
) -> dict[str, Any]:
    try:
        remote = wandb.Api(timeout=60).run(locator)
    except Exception as error:
        if internal_step == 0 and _not_found(error):
            return {"identity": list(identity), "result": "run_id_absent"}
        raise
    if internal_step == 0:
        raise RuntimeError("fresh step-0 W&B run ID already exists")
    if _remote_identity(remote) != identity:
        raise RuntimeError("W&B predecessor identity differs from the checkpoint lineage")
    last_step = int(getattr(remote, "lastHistoryStep", -1))
    if last_step != internal_step - 1:
        raise RuntimeError(
            f"W&B predecessor is at History step {last_step}, expected {internal_step - 1}"
        )
    return {"identity": list(identity), "last_history_step": last_step}


def _retryable_api_error(error: Exception) -> bool:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and (status in (408, 429) or status >= 500):
        return True
    return type(error).__name__ in {"ConnectTimeout", "ConnectionError", "ReadTimeout", "Timeout"}


def _wait_for_publication(
    wandb: Any,
    *,
    locator: str,
    identity: tuple[str, str, str],
    internal_step: int,
    optimizer_step: int,
    metrics: dict[str, float],
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_mismatch = "W&B publication has not propagated"
    while True:
        try:
            remote = wandb.Api(timeout=60).run(locator)
            rows = list(
                remote.scan_history(
                    keys=["_step", STEP_METRIC, *VALIDATION_METRICS],
                    min_step=internal_step,
                    max_step=internal_step + 1,
                    page_size=10,
                )
            )
            exact_rows = []
            for row in rows:
                try:
                    exact = (
                        int(row.get("_step", -1)) == internal_step
                        and int(row.get(STEP_METRIC, -1)) == optimizer_step
                        and all(
                            math.isclose(_finite_float(row.get(key), key), value, rel_tol=0.0, abs_tol=1e-12)
                            for key, value in metrics.items()
                        )
                    )
                except (TypeError, ValueError):
                    exact = False
                if exact:
                    exact_rows.append(dict(row))
            summary = dict(remote.summary)
            try:
                summary_exact = all(
                    math.isclose(
                        _finite_float(summary.get(key), f"summary {key}"),
                        value,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for key, value in metrics.items()
                )
            except ValueError as error:
                summary_exact = False
                last_mismatch = str(error)
            index_entry = _summary_index(summary).get(str(optimizer_step))
            index_exact = index_entry is not None
            remote_identity = _remote_identity(remote)
            if remote_identity != identity:
                raise RuntimeError(f"W&B publication identity mismatch: {remote_identity!r}")
            if (
                int(getattr(remote, "lastHistoryStep", -1)) == internal_step
                and len(exact_rows) == 1
                and summary_exact
                and index_exact
            ):
                return {
                    "identity": list(identity),
                    "last_history_step": internal_step,
                    "history_row": exact_rows[0],
                    "summary_metrics": {key: summary[key] for key in VALIDATION_METRICS},
                    "summary_index_step": index_entry,
                }
            last_mismatch = (
                "W&B publication is incomplete: "
                f"lastHistoryStep={getattr(remote, 'lastHistoryStep', None)!r}, "
                f"exact_rows={len(exact_rows)}, summary_exact={summary_exact}, index_exact={index_exact}"
            )
        except Exception as error:
            if not _retryable_api_error(error):
                raise
            last_mismatch = f"transient W&B API error: {type(error).__name__}: {error}"
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"fresh W&B validation History/Summary readback did not converge: {last_mismatch}"
            )
        time.sleep(5)


def publish(args: argparse.Namespace) -> dict[str, Any]:
    receipt, optimizer_step, metrics = load_evaluation_receipt(
        args.evaluation_receipt,
        expected_run_id=args.run_id,
        expected_name=args.name,
    )
    internal_step = validation_internal_step(optimizer_step)
    timeout_seconds = int(args.timeout_seconds)
    if timeout_seconds < 1 or timeout_seconds > 600:
        raise ValueError("readback timeout must be in [1,600] seconds")

    import wandb

    identity = (args.run_id, args.entity, args.project)
    locator = f"{args.entity}/{args.project}/{args.run_id}"
    predecessor = _read_predecessor(
        wandb,
        locator=locator,
        identity=identity,
        internal_step=internal_step,
    )
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.name if optimizer_step == 0 else None,  # resume keeps the trainer-set name
        resume="never" if optimizer_step == 0 else "must",
    )
    try:
        if _remote_identity(run) != identity:
            raise RuntimeError("active W&B writer identity differs from the checkpoint lineage")
        run.define_metric(STEP_METRIC)
        for metric in VALIDATION_METRICS:
            run.define_metric(metric, step_metric=STEP_METRIC)
        run.define_metric(f"{AUDIT_PREFIX}/*", hidden=True)
        row = {
            STEP_METRIC: optimizer_step,
            **metrics,
            f"{AUDIT_PREFIX}/user_step": optimizer_step,
            f"{AUDIT_PREFIX}/optimizer_step": optimizer_step,
        }
        run.log(row, step=internal_step, commit=True)
        for metric, value in metrics.items():
            run.summary[metric] = value
        index_key = f"{AUDIT_PREFIX}_index"
        summary_index = _summary_index(run.summary)
        summary_index[str(optimizer_step)] = {
            "internal_step": internal_step,
            "checkpoint": receipt["checkpoint"],
        }
        run.summary[index_key] = summary_index
    finally:
        run.finish(exit_code=0)

    readback = _wait_for_publication(
        wandb,
        locator=locator,
        identity=identity,
        internal_step=internal_step,
        optimizer_step=optimizer_step,
        metrics=metrics,
        timeout_seconds=timeout_seconds,
    )
    document = {
        "schema_version": 1,
        "status": "accepted",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": args.slurm_job_id,
        "parent_wandb_run_id": args.run_id,
        "parent_wandb_name": args.name,
        "wandb_entity": args.entity,
        "wandb_project": args.project,
        "wandb_internal_step": internal_step,
        "user_step": optimizer_step,
        "optimizer_step": optimizer_step,
        "step_metric": STEP_METRIC,
        "metrics": metrics,
        "checkpoint": receipt["checkpoint"],
        "evaluation_receipt": str(args.evaluation_receipt.resolve()),
        "predecessor_readback": predecessor,
        "fresh_remote_readback": readback,
        "visible_validation_group": "val-core",
        "hidden_audit_prefix": AUDIT_PREFIX,
    }
    _atomic_json(args.publication_receipt.resolve(), document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--evaluation-receipt", type=Path, required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument("--slurm-job-id", default=os.environ.get("SLURM_JOB_ID"), required=False)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if not args.slurm_job_id:
        raise SystemExit("--slurm-job-id or SLURM_JOB_ID is required")
    print(json.dumps(publish(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
