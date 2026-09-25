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

import multiprocessing
import signal
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Optional

try:
    from math_verify.errors import TimeoutException
except ImportError:

    class TimeoutException(Exception):
        pass

    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

_pool = None
_pool_lock = threading.Lock()
_batch_verify_func = None


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> float:
    """Run math_verify in a subprocess where signal.alarm() works."""
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(model_output, pred_targets)
    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
        ret_score = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in math_verify compute_score: {e}")
    return ret_score


@contextmanager
def _time_limit(seconds: int):
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError("math_verify scoring timed out")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the final boxed answer from the last part of a model response."""
    if not isinstance(text, str):
        return None

    tokens = text.split()
    search_text = text if len(tokens) <= 300 else " ".join(tokens[-300:])
    needle = r"\boxed{"
    start = search_text.rfind(needle)
    if start == -1:
        return None

    index = start + len(needle)
    depth = 1
    while index < len(search_text):
        char = search_text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return search_text[start + len(needle) : index].strip()
        index += 1
    return None


def _ensure_batch_verify_func():
    global _batch_verify_func
    if _batch_verify_func is None:
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        _batch_verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
    return _batch_verify_func


def _compute_boxed_score(model_output: str, ground_truth: str, timeout_score: float = 0.0) -> float:
    boxed_answer = extract_boxed_answer(model_output)
    if not boxed_answer:
        return float(timeout_score)

    verify_func = _ensure_batch_verify_func()
    try:
        score, _ = verify_func([f"\\boxed{{{ground_truth}}}"], [f"\\boxed{{{boxed_answer}}}"])
    except TimeoutException:
        return float(timeout_score)
    except Exception:
        return float(timeout_score)
    return 1.0 if float(score) == 1.0 else 0.0


def compute_boxed_score_bounded(
    model_output: str,
    ground_truth: str,
    *,
    timeout: float = 20.0,
    timeout_score: float = 0.0,
) -> float:
    """Bound synchronous batch scoring even when math-verify fails to return."""
    try:
        with _time_limit(max(1, int(timeout))):
            return float(_compute_boxed_score(model_output, ground_truth, timeout_score=timeout_score))
    except (TimeoutError, TimeoutException):
        return float(timeout_score)
    except Exception:
        return float(timeout_score)


def _parallel_score_worker(
    model_output: str,
    ground_truth: str,
    timeout_score: float,
    inner_timeout_s: int,
) -> float:
    try:
        with _time_limit(inner_timeout_s):
            return float(_compute_boxed_score(model_output, ground_truth, timeout_score=timeout_score))
    except Exception as exc:
        warnings.warn(f"math_verify worker error: {exc}")
        return float(timeout_score)


def parallel_compute_score(
    model_outputs: list[str],
    ground_truths: list[str],
    timeout: int = 20,
    num_processes: int = 32,
    *,
    timeout_score: float = 0.0,
    maxtasksperchild: int = 200,
    start_method_priority: tuple[str, ...] = ("forkserver", "spawn"),
) -> list[float]:
    """Score a batch of math responses with process-level timeout protection."""
    if len(model_outputs) != len(ground_truths):
        raise ValueError("model_outputs and ground_truths must have the same length")
    if not model_outputs:
        return []

    context = None
    for method in start_method_priority:
        try:
            context = multiprocessing.get_context(method)
            break
        except ValueError:
            continue
    if context is None:
        context = multiprocessing.get_context()

    inner_timeout_s = max(1, int(timeout) - 1)
    worker = partial(
        _parallel_score_worker,
        timeout_score=timeout_score,
        inner_timeout_s=inner_timeout_s,
    )
    results: list[float | None] = [None] * len(model_outputs)

    while any(result is None for result in results):
        remaining = [
            (index, (model_outputs[index], ground_truths[index]))
            for index, result in enumerate(results)
            if result is None
        ]
        if not remaining:
            break

        process_count = max(1, min(int(num_processes), len(remaining)))
        pool = context.Pool(processes=process_count, maxtasksperchild=maxtasksperchild)
        restart_pool = False
        try:
            async_results = {
                index: pool.apply_async(worker, args=args)
                for index, args in remaining
            }
            for index, async_result in async_results.items():
                try:
                    results[index] = float(async_result.get(timeout=timeout))
                except multiprocessing.TimeoutError:
                    warnings.warn(f"math_verify timeout on batch index {index}")
                    results[index] = float(timeout_score)
                    restart_pool = True
                    break
                except Exception as exc:
                    warnings.warn(f"math_verify result error on batch index {index}: {exc}")
                    results[index] = float(timeout_score)

            if restart_pool:
                pool.terminate()
                pool.join()
                continue

            pool.close()
            pool.join()
        finally:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass

    return [float(timeout_score) if result is None else float(result) for result in results]
