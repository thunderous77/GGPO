# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
from math import nan

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    ActorConfig,
    FSDPActorConfig,
    McoreActorConfig,
    OptimizerConfig,
    PolicyLossConfig,
)


class TestActorConfig(unittest.TestCase):
    """Test the ActorConfig dataclass and its variants."""

    def _valid_ggpo_actor_config(self, **overrides):
        """Build a valid GGPO ActorConfig; overrides should isolate one invalid behavior."""
        config_kwargs = {
            "strategy": "fsdp",
            "use_dynamic_bsz": False,
            "ppo_mini_batch_size": 8,
            "ppo_micro_batch_size_per_gpu": 16,
            "policy_loss": PolicyLossConfig(loss_mode="ggpo", beta=0.01),
            "use_kl_loss": False,
            "shuffle": False,
            "rollout_n": 8,
        }
        config_kwargs.update(overrides)
        return ActorConfig(**config_kwargs)

    def test_config_inheritance(self):
        """Test that the inheritance hierarchy works correctly."""
        megatron_dict = {
            "_target_": "verl.workers.config.McoreActorConfig",
            "strategy": "megatron",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 256,
            "clip_ratio": 0.2,
            "optim": {
                "_target_": "verl.workers.config.McoreOptimizerConfig",
                "lr": 0.1,
            },
            "rollout_n": 1,
        }
        fsdp_dict = {
            "_target_": "verl.workers.config.FSDPActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 256,
            "clip_ratio": 0.2,
            "optim": {
                "_target_": "verl.workers.config.FSDPOptimizerConfig",
                "lr": 0.1,
            },
            "rollout_n": 1,
        }

        megatron_config = omega_conf_to_dataclass(megatron_dict)
        fsdp_config = omega_conf_to_dataclass(fsdp_dict)

        self.assertIsInstance(megatron_config, ActorConfig)
        self.assertIsInstance(fsdp_config, ActorConfig)

        self.assertEqual(megatron_config.ppo_mini_batch_size, fsdp_config.ppo_mini_batch_size)
        self.assertEqual(megatron_config.clip_ratio, fsdp_config.clip_ratio)

    def test_actor_config_from_yaml(self):
        """Test creating ActorConfig from YAML file."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(config_name="actor", overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=128"])

        config = omega_conf_to_dataclass(cfg)

        self.assertIsInstance(config, ActorConfig)
        self.assertEqual(config.strategy, "fsdp")
        self.assertFalse(hasattr(config.policy_loss, "objective"))
        self.assertFalse(hasattr(config.policy_loss, "objective_normalization"))

    def test_policy_loss_ggpo_defaults(self):
        """GGPO policy loss fields default to the documented hyperparameters."""
        config = PolicyLossConfig(loss_mode="ggpo")

        self.assertEqual(config.beta, 0.01)
        self.assertFalse(hasattr(config, "group_size"))
        self.assertFalse(hasattr(config, "objective"))
        self.assertFalse(hasattr(config, "objective_normalization"))
        self.assertFalse(hasattr(config, "negative_reward"))
        self.assertFalse(hasattr(config, "positive_reward"))
        self.assertFalse(hasattr(config, "require_binary_rewards"))
        self.assertEqual(config.sequence_log_prob_reduction, "sum")

    def test_entropy_switch_defaults_to_existing_enabled_behavior(self):
        self.assertTrue(self._valid_ggpo_actor_config().calculate_entropy)

    def test_nonzero_entropy_coefficient_requires_entropy_switch(self):
        with self.assertRaisesRegex(ValueError, "calculate_entropy"):
            self._valid_ggpo_actor_config(entropy_coeff=0.1, calculate_entropy=False)

    def test_policy_loss_ggpo_explicit_values(self):
        """GGPO policy loss fields can be explicitly configured."""
        config = PolicyLossConfig(
            loss_mode="ggpo",
            beta=0.5,
            sequence_log_prob_reduction="mean",
        )

        self.assertEqual(config.beta, 0.5)
        self.assertEqual(config.sequence_log_prob_reduction, "mean")

    def test_fsdp_actor_config_from_yaml(self):
        """Test creating FSDPActorConfig from YAML file."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(config_name="dp_actor", overrides=["strategy=fsdp2", "ppo_micro_batch_size_per_gpu=128"])

        config = omega_conf_to_dataclass(cfg)

        self.assertIsInstance(config, FSDPActorConfig)
        self.assertEqual(config.strategy, "fsdp2")

    def test_megatron_actor_config_from_yaml(self):
        """Test creating McoreActorConfig from YAML file."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
            cfg = compose(config_name="megatron_actor", overrides=["ppo_micro_batch_size_per_gpu=128"])

        config = omega_conf_to_dataclass(cfg)

        self.assertIsInstance(config, McoreActorConfig)
        self.assertEqual(config.strategy, "megatron")

    def test_config_get_method(self):
        """Test the get method for backward compatibility."""
        config_dict = {
            "_target_": "verl.workers.config.ActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 256,
            "optim": {
                "_target_": "verl.workers.config.OptimizerConfig",
                "lr": 0.1,
            },
            "rollout_n": 1,
        }
        config = omega_conf_to_dataclass(config_dict)

        self.assertEqual(config.get("strategy"), "fsdp")
        self.assertEqual(config.get("ppo_mini_batch_size"), 256)

        self.assertIsNone(config.get("non_existing"))
        self.assertEqual(config.get("non_existing", "default"), "default")

    def test_config_dict_like_access(self):
        """Test dictionary-like access to config fields."""
        config_dict = {
            "_target_": "verl.workers.config.ActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 256,
            "optim": {
                "_target_": "verl.workers.config.OptimizerConfig",
                "lr": 0.1,
            },
            "rollout_n": 1,
        }
        config = omega_conf_to_dataclass(config_dict)

        self.assertEqual(config["strategy"], "fsdp")
        self.assertEqual(config["ppo_mini_batch_size"], 256)

        field_names = list(config)
        self.assertIn("strategy", field_names)
        self.assertIn("ppo_mini_batch_size", field_names)

        self.assertGreater(len(config), 0)

    def test_frozen_fields_modification_raises_exception(self):
        """Test that modifying frozen fields raises an exception."""
        config_dict = {
            "_target_": "verl.workers.config.ActorConfig",
            "strategy": "fsdp",
            "ppo_mini_batch_size": 256,
            "ppo_micro_batch_size_per_gpu": 256,
            "optim": {
                "_target_": "verl.workers.config.OptimizerConfig",
                "lr": 0.1,
            },
            "rollout_n": 1,
        }
        config = omega_conf_to_dataclass(config_dict)

        with self.assertRaises(AttributeError):
            config.strategy = "megatron"

        with self.assertRaises(AttributeError):
            config.clip_ratio = 0.5

        config.ppo_mini_batch_size = 512  # This should work since it's not in frozen fields anymore
        self.assertEqual(config.ppo_mini_batch_size, 512)

    def test_actor_config_validation_exceptions(self):
        """Test that ActorConfig.__post_init__ raises appropriate validation exceptions."""
        optim = OptimizerConfig(lr=0.1)
        with self.assertRaises((ValueError, AssertionError)) as cm:
            ActorConfig(
                strategy="fsdp",
                loss_agg_mode="invalid-mode",
                use_dynamic_bsz=True,
                optim=optim,
                ppo_micro_batch_size_per_gpu=4,
                rollout_n=1,
            )
        self.assertIn("Invalid loss_agg_mode", str(cm.exception))

        with self.assertRaises((ValueError, AssertionError)) as cm:
            ActorConfig(
                strategy="fsdp",
                use_dynamic_bsz=False,
                ppo_micro_batch_size=4,
                ppo_micro_batch_size_per_gpu=2,
                optim=optim,
                rollout_n=1,
            )
        self.assertIn("You have set both", str(cm.exception))

        with self.assertRaises((ValueError, AssertionError)) as cm:
            ActorConfig(
                strategy="fsdp",
                use_dynamic_bsz=False,
                ppo_micro_batch_size=None,
                ppo_micro_batch_size_per_gpu=None,
                optim=optim,
                rollout_n=1,
            )
        self.assertIn("Please set at least one", str(cm.exception))

        config = ActorConfig(
            strategy="fsdp",
            use_dynamic_bsz=True,
            ppo_micro_batch_size=None,
            ppo_micro_batch_size_per_gpu=None,
            optim=optim,
            rollout_n=1,
        )
        self.assertIsNotNone(config)  # Should not raise an exception

    def test_fsdp_actor_config_validation_exceptions(self):
        """Test that FSDPActorConfig.validate() raises appropriate validation exceptions."""
        optim = OptimizerConfig(lr=0.1)
        config = FSDPActorConfig(
            strategy="fsdp",
            ulysses_sequence_parallel_size=2,
            use_dynamic_bsz=True,  # Skip batch size validation to focus on FSDP validation
            optim=optim,
            rollout_n=1,
        )

        model_config = {"use_remove_padding": False}
        with self.assertRaises(ValueError) as cm:
            config.validate(n_gpus=8, train_batch_size=256, model_config=model_config)
        self.assertIn("you must enable `use_remove_padding`", str(cm.exception))

    def test_actor_config_validate_method_exceptions(self):
        """Test that ActorConfig.validate() raises appropriate validation exceptions."""
        optim = OptimizerConfig(lr=0.1)
        config = ActorConfig(
            strategy="fsdp",
            use_dynamic_bsz=False,
            ppo_mini_batch_size=256,
            ppo_micro_batch_size=8,
            ppo_micro_batch_size_per_gpu=None,  # Ensure only one batch size setting is used
            optim=optim,
            rollout_n=1,
        )

        with self.assertRaises(ValueError) as cm:
            config.validate(n_gpus=8, train_batch_size=128)
        self.assertIn("train_batch_size", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            config.validate(n_gpus=16, train_batch_size=512)
        self.assertIn("must be >= n_gpus", str(cm.exception))

    def test_ggpo_beta_must_be_finite_and_positive(self):
        """GGPO beta must be finite and strictly positive."""
        for beta in (0.0, -0.1, nan):
            with self.subTest(beta=beta):
                config = self._valid_ggpo_actor_config(
                    policy_loss=PolicyLossConfig(loss_mode="ggpo", beta=beta)
                )

                with self.assertRaisesRegex(ValueError, "beta"):
                    config.validate(n_gpus=1, train_batch_size=8)

    def test_agro_accepts_nondefault_positive_beta(self):
        """AGRO beta is a tunable positive coefficient, not a fixed experiment constant."""
        config = self._valid_ggpo_actor_config(
            policy_loss=PolicyLossConfig(loss_mode="agro", beta=0.25)
        )

        config.validate(n_gpus=1, train_batch_size=8)

    def test_ggpo_rollout_n_must_be_greater_than_one(self):
        """GGPO derives its group size from rollout_n, which must exceed one."""
        config = self._valid_ggpo_actor_config(
            policy_loss=PolicyLossConfig(loss_mode="ggpo", beta=0.01),
            rollout_n=1,
        )

        with self.assertRaisesRegex(ValueError, "rollout_n.*> 1"):
            config.validate(n_gpus=1, train_batch_size=8)

    def test_removed_ggpo_objective_constructor_arguments_are_rejected(self):
        """Removed GGPO selectors cannot be reached through the dataclass API."""
        removed_arguments = {
            "group_size": 8,
            "objective": "variance",
            "objective_normalization": "beta_sqrt_group_minus_one",
        }
        for name, value in removed_arguments.items():
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, name):
                PolicyLossConfig(loss_mode="ggpo", beta=0.01, **{name: value})

    def test_ggpo_sequence_log_prob_reduction_must_be_supported(self):
        """GGPO sequence log-probability reduction is either sum or mean."""
        config = self._valid_ggpo_actor_config(
            policy_loss=PolicyLossConfig(loss_mode="ggpo", sequence_log_prob_reduction="token_mean")
        )

        with self.assertRaisesRegex(ValueError, "sequence_log_prob_reduction"):
            config.validate(n_gpus=1, train_batch_size=8)

    def test_ggpo_use_kl_loss_must_be_false(self):
        """GGPO embeds reference regularization and must not also enable KL loss."""
        config = self._valid_ggpo_actor_config(use_kl_loss=True)

        with self.assertRaisesRegex(ValueError, "use_kl_loss"):
            config.validate(n_gpus=1, train_batch_size=8)

    def test_ggpo_use_dynamic_bsz_preserves_grouped_execution(self):
        """GGPO may use dynamic batching because actor updates force complete groups."""
        config = self._valid_ggpo_actor_config(use_dynamic_bsz=True)

        config.validate(n_gpus=1, train_batch_size=8)

    def test_ggpo_shuffle_must_be_false(self):
        """GGPO requires deterministic group ordering."""
        config = self._valid_ggpo_actor_config(shuffle=True)

        with self.assertRaisesRegex(ValueError, "shuffle"):
            config.validate(n_gpus=1, train_batch_size=8)

    def test_asympo_does_not_restrict_actor_shuffle(self):
        """ASymPO is response-local and does not require contiguous policy groups."""
        config = self._valid_ggpo_actor_config(
            policy_loss=PolicyLossConfig(loss_mode="asympo"),
            shuffle=True,
            use_kl_loss=False,
        )

        config.validate(n_gpus=1, train_batch_size=8)


if __name__ == "__main__":
    unittest.main()
