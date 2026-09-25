"""Safety checks for AReaL's destructive Hugging Face disk loader."""

from pathlib import Path
from typing import Any

from .factory import config_get


def areal_weight_update_key(experiment_name: str, trial_name: str, controller_version: int) -> str:
    """Build AReaL's disk-update rendezvous key for the currently loaded version."""

    controller_version = int(controller_version)
    if controller_version < 0:
        raise ValueError(f"AReaL controller version must be non-negative, got {controller_version}")

    # Keep AReaL lazy for native VERL callers of the path helpers below.
    from .vendored import load_vendored_areal

    load_vendored_areal()
    from areal.utils import names

    return names.update_weights_from_disk(experiment_name, trial_name, controller_version)


def transient_weight_export_root(config: Any) -> Path:
    """Resolve the same private export root used by ``FullyAsyncTrainer``."""

    configured_root = config_get(config, "async_training.weight_sync.export_dir", None)
    if configured_root is None:
        checkpoint_root = config_get(config, "trainer.default_local_dir", None)
        if not checkpoint_root:
            raise ValueError("trainer.default_local_dir is required when weight_sync.export_dir is null")
        configured_root = Path(str(checkpoint_root)) / "async_weight_exports"
    return Path(str(configured_root)).expanduser().resolve()


def validate_transient_hf_export(config: Any, *, version: int, hf_path: str) -> Path:
    """Allow only ``<private-root>/version_N/huggingface`` to be consumed.

    Vendored AReaL removes the supplied directory after loading it, regardless
    of ``WeightUpdateMeta.clear_checkpoint_after_load``.  An exact path check
    prevents a malformed caller from deleting a source model or checkpoint.
    """

    version = int(version)
    if version < 0:
        raise ValueError(f"AReaL weight version must be non-negative, got {version}")
    export_path = Path(hf_path).expanduser().resolve()
    expected = (transient_weight_export_root(config) / f"version_{version}" / "huggingface").resolve()
    if export_path != expected:
        raise ValueError(
            "Refusing destructive AReaL disk load outside the exact transient export path: "
            f"expected {expected}, got {export_path}"
        )
    if not export_path.is_dir() or not (export_path / "config.json").is_file():
        raise ValueError(f"AReaL weight sync requires a complete HF export with config.json: {export_path}")
    return export_path
