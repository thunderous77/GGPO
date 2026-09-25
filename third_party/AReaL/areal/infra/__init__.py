# SPDX-License-Identifier: Apache-2.0

"""Core components for AREAL."""

from . import workflow_context
from .controller import RolloutController, TrainController
from .platforms import Platform, current_platform, is_npu_available
from .remote_inf_engine import (
    RemoteInfBackendProtocol,
    RemoteInfEngine,
)
from .scheduler import LocalScheduler, RayScheduler, SlurmScheduler
from .staleness_manager import StalenessManager
from .workflow_executor import (
    WorkflowExecutor,
    check_trajectory_format,
)


def __getattr__(name: str):
    """Load optional launchers only when callers request them.

    The VERL rollout adapter uses AReaL's controllers and schedulers, but not
    its training launchers.  Keeping launcher imports lazy also lets that
    adapter run with VERL's supported torch build, where newer optional DCP
    staging symbols used only by AReaL checkpointing are unavailable.
    """
    launcher_names = {
        "LocalLauncher",
        "RayLauncher",
        "SGLangServerWrapper",
        "SlurmLauncher",
        "vLLMServerWrapper",
    }
    if name in launcher_names:
        from .launcher import (
            LocalLauncher,
            RayLauncher,
            SGLangServerWrapper,
            SlurmLauncher,
            vLLMServerWrapper,
        )

        launchers = {
            "LocalLauncher": LocalLauncher,
            "RayLauncher": RayLauncher,
            "SGLangServerWrapper": SGLangServerWrapper,
            "SlurmLauncher": SlurmLauncher,
            "vLLMServerWrapper": vLLMServerWrapper,
        }
        globals().update(launchers)
        return launchers[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RemoteInfBackendProtocol",
    "RemoteInfEngine",
    "StalenessManager",
    "WorkflowExecutor",
    "check_trajectory_format",
    "RolloutController",
    "TrainController",
    "workflow_context",
    "Platform",
    "current_platform",
    "is_npu_available",
    "LocalScheduler",
    "RayScheduler",
    "SlurmScheduler",
    "LocalLauncher",
    "RayLauncher",
    "SlurmLauncher",
    "SGLangServerWrapper",
    "vLLMServerWrapper",
]
