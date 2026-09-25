"""Pluggable rollout backends for fully asynchronous policy optimization.

The package intentionally avoids importing the AReaL actor eagerly.  Native
VERL users therefore do not need AReaL's optional runtime dependencies unless
``async_training.rollout_backend=areal`` is selected.
"""

from .capabilities import AREAL_CAPABILITIES, VERL_CAPABILITIES, RolloutBackendCapabilities
from .factory import create_rollout_backend, get_rollout_backend_actor_class, resolve_rollout_backend_name

__all__ = [
    "AREAL_CAPABILITIES",
    "VERL_CAPABILITIES",
    "RolloutBackendCapabilities",
    "create_rollout_backend",
    "get_rollout_backend_actor_class",
    "resolve_rollout_backend_name",
]
