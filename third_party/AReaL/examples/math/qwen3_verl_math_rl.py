"""Run asynchronous Qwen3 GRPO on a VERL-format math parquet.

Set ``train_dataset.dataset_kwargs.dataset_name=verl_math`` (and the same for
``valid_dataset`` when present) so dataset selection is explicit and does not
depend on a substring in the parquet path.
"""

import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            split="test",
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )

    workflow_kwargs = {
        "reward_fn": "areal.reward.verl_math.verl_math_reward_fn",
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "enable_thinking": True,
    }
    eval_workflow_kwargs = None
    if valid_dataset is not None:
        eval_workflow_kwargs = dict(workflow_kwargs)
        eval_workflow_kwargs["gconfig"] = config.eval_gconfig

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow=(
                "areal.workflow.rlvr.RLVRWorkflow"
                if valid_dataset is not None
                else None
            ),
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
