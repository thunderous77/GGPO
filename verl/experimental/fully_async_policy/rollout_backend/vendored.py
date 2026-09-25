"""Strict loader for the repository-vendored AReaL package."""

import importlib
import sys
from pathlib import Path
from types import ModuleType


def vendored_areal_root() -> Path:
    root = Path(__file__).resolve().parents[4] / "third_party" / "AReaL"
    package_init = root / "areal" / "__init__.py"
    if not package_init.is_file():
        raise ImportError(
            "The AReaL rollout backend requires the vendored source at "
            f"{package_init}; no external AREAL_ROOT fallback is permitted"
        )
    return root.resolve()


def _is_beneath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _verify_module(module: ModuleType, root: Path) -> ModuleType:
    module_file = getattr(module, "__file__", None)
    if module_file is None or not _is_beneath(Path(module_file), root):
        raise ImportError(
            "Refusing to use a non-vendored 'areal' package. Expected imports beneath "
            f"{root}, got {module_file!r}"
        )
    return module


def load_vendored_areal() -> ModuleType:
    """Import AReaL from ``third_party/AReaL`` and verify provenance."""

    root = vendored_areal_root()
    loaded = sys.modules.get("areal")
    if loaded is not None:
        return _verify_module(loaded, root)

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return _verify_module(importlib.import_module("areal"), root)
