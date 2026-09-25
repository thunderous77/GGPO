"""Capability descriptions shared by the fully-async rollout actors."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RolloutBackendCapabilities:
    """Operations exposed to the backend-neutral trainer."""

    name: str
    weight_sync: str
    profiling: bool
    hybrid_validation: bool
    hybrid_scaling: bool
    atomic_prompt_groups: bool

    def as_dict(self) -> dict[str, str | bool]:
        return asdict(self)


VERL_CAPABILITIES = RolloutBackendCapabilities(
    name="verl",
    weight_sync="verl_checkpoint_engine",
    profiling=True,
    hybrid_validation=True,
    hybrid_scaling=True,
    atomic_prompt_groups=True,
)

AREAL_CAPABILITIES = RolloutBackendCapabilities(
    name="areal",
    weight_sync="hf_disk",
    profiling=False,
    hybrid_validation=False,
    hybrid_scaling=False,
    atomic_prompt_groups=True,
)
