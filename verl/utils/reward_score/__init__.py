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
# from . import gsm8k, math, prime_math, prime_code

import os

from verl.utils.import_utils import deprecated

MATH_VERIFY_DATA_SOURCES = {
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
    "HuggingFaceH4/MATH-500",
    "aime24",
    "aime25",
    "math500",
    "dapo_filter",
    "openr1_filtered",
    "guru_rl92k_math",
    "merged_openr1_guru",
    "alp_openr1_filtered",
    "alp_guru_rl92k_math",
    "alp_merged_openr1_guru",
    "alp_math500",
}


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in MATH_VERIFY_DATA_SOURCES:
        from . import math_verify

        res = math_verify.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


def batch_compute_score(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos=None,
    timeout: int = 20,
    num_processes: int = 32,
    timeout_score: float = 0.0,
    **kwargs,
):
    """Compute math rewards for a batch, falling back per sample for mixed sources."""
    data_sources = list(data_sources)
    solution_strs = list(solution_strs)
    ground_truths = list(ground_truths)

    if len(solution_strs) != len(ground_truths) or len(solution_strs) != len(data_sources):
        raise ValueError("data_sources, solution_strs, and ground_truths must have the same length")

    if all(data_source in MATH_VERIFY_DATA_SOURCES for data_source in data_sources):
        from . import math_verify

        scores = [float(timeout_score)] * len(solution_strs)
        parser_success = [0.0] * len(solution_strs)
        boxed_count = 0
        for index, (solution_str, ground_truth) in enumerate(zip(solution_strs, ground_truths, strict=True)):
            if math_verify.extract_boxed_answer(solution_str):
                boxed_count += 1
                parser_success[index] = 1.0
                scores[index] = float(
                    math_verify.compute_boxed_score_bounded(
                        solution_str,
                        ground_truth,
                        timeout=timeout,
                        timeout_score=timeout_score,
                    )
                )
        if os.getenv("VERL_BATCH_REWARD_DEBUG") == "1":
            positives = sum(score > 0 for score in scores)
            mean_score = sum(scores) / len(scores) if scores else 0.0
            print(
                "[batch_compute_score] "
                f"module={__file__} n={len(scores)} boxed={boxed_count} "
                f"positives={positives} mean={mean_score} "
                f"sources={sorted({str(source) for source in data_sources})[:5]}",
                flush=True,
            )
        if os.getenv("VERL_GROUP_REWARD_DEBUG") == "1":
            try:
                group_size = int(os.getenv("VERL_GROUP_REWARD_DEBUG_GROUP_SIZE", "8"))
            except ValueError:
                group_size = 8
            if group_size > 1 and len(scores) >= group_size:
                groups = [scores[i : i + group_size] for i in range(0, len(scores), group_size)]
                complete_groups = [group for group in groups if len(group) == group_size]
                sums = [sum(group) for group in complete_groups]
                all_zero = sum(total == 0 for total in sums)
                all_one = sum(total == group_size for total in sums)
                mixed = len(sums) - all_zero - all_one
                try:
                    sample_limit = int(os.getenv("VERL_GROUP_REWARD_DEBUG_SAMPLE_GROUPS", "6"))
                except ValueError:
                    sample_limit = 6
                sample_limit = max(0, sample_limit)
                samples = [
                    [int(score) if score in (0.0, 1.0) else score for score in group]
                    for group in complete_groups[:sample_limit]
                ]
                print(
                    "[batch_group_rewards] "
                    f"group_size={group_size} groups={len(sums)} all_zero={all_zero} "
                    f"mixed={mixed} all_one={all_one} "
                    f"sum_min={min(sums) if sums else 0} sum_max={max(sums) if sums else 0} "
                    f"sum_mean={sum(sums) / len(sums) if sums else 0.0} samples={samples}",
                    flush=True,
                )
        if os.getenv("VERL_PAIRED_SMOKE_DIAGNOSTICS") == "1":
            return [
                {"score": score, "acc": score, "parser_success": parsed}
                for score, parsed in zip(scores, parser_success, strict=True)
            ]
        return scores

    extra_infos = [{} for _ in solution_strs] if extra_infos is None else list(extra_infos)
    return [
        default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        )
    ]


def get_default_compute_score(reward_name: str | None):
    """Get the default compute_score function based on the reward manager type."""
    if reward_name == "batch":
        return batch_compute_score
    return default_compute_score


__all__ = ["batch_compute_score", "default_compute_score", "get_default_compute_score"]
