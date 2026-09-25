# SPDX-License-Identifier: Apache-2.0
"""Reward function whose score is byte-identical to verl's
``verl.utils.reward_score.math_verify.compute_score``. Use for eval to make AReaL
numbers directly comparable to a verl baseline.

Differences from ``areal.reward.gsm8k.gsm8k_reward_fn``:

- gold is wrapped in ``\\boxed{}`` before parsing; gold parsed with
  ``LatexExtractionConfig`` only.
- pred parsed with ``(ExprExtractionConfig(), LatexExtractionConfig())`` — i.e.
  ``try_extract_without_anchor=False`` (default), unlike gsm8k_reward_fn which
  uses ``try_extract_without_anchor=True`` on both gold and pred.
- 30 s timeout (vs 5 s in MathVerifyWorker) — long LaTeX expressions in MATH-style
  responses routinely exceed 5 s and were being silently scored 0.
- Wraps each call in a per-invocation ``ThreadPoolExecutor`` with
  ``shutdown(wait=False)`` on timeout. We previously tried math_verify's built-in
  ``parsing_timeout`` (SIGALRM) to interrupt stuck sympy without leaking, but it
  silently returned empty parses inside the ``ProcessPoolExecutor`` worker
  subprocess — every reward came back 0 from step 0 of run pair 5. Reverted to
  the thread wrapper but use ``shutdown(wait=False)`` on timeout so we don't
  block on the stuck thread (the previous ``with executor`` form did wait, and
  that's where the 2026-06-02 OOM came from).
"""

import concurrent.futures
import resource

from math_verify.grader import verify as math_verify_verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

# Memory-leak investigation 2026-06-07 — see scripts/diag-notes.md.
#
# Tested hypotheses + results on synthetic + real JSONL data:
#  - SYMPY_GROUND_TYPES=python (no gmpy2): no effect.
#  - sympy.cache.clear_cache() per iter: no effect.
#  - Clear ANTLR4 class-level DFA + sharedContextCache per iter: no effect.
#  - parse_latex_cached / parse_expr_cached cache_clear per iter: HARMFUL —
#    real data has 72% cache-hit rate; clearing forces re-parse of repeated
#    inputs, increasing allocation pressure.
#  - malloc_trim(0) + gc.collect() per iter: ~33% RSS reduction. Helps.
#
# The real-data repro (worker_*.jsonl × 1500 calls) caps at ~77 MB worker
# RSS — bounded. Production 200-500 GB OOMs must come from pathological
# inputs absent from my data sample (extremely deep nesting from late-
# training thinking traces). Synthetic depth-2000 nested-paren expressions
# allocate 38 MB per parse and can take 135 s — exceeding the 15 s timeout
# and abandoning a thread that keeps grinding.
#
# Defenses that DO work, layered:
#  1. Input pre-check: skip math_verify entirely on obviously-pathological
#     inputs (very long LaTeX or extreme nesting). Avoids the worst cases.
#  2. max_tasks_per_child=20 in ProcessPoolExecutor (reward_api.py):
#     bounded worker lifetime.
#  3. Per-call RSS check + suicide at 6 GB (this file): catches outliers.
#  4. ANTLR / sympy cache management is NOT a fix (verified above).
#
# This module keeps the cleanup hooks for safety but the primary defense
# is the input pre-check below.
try:
    from latex2sympy2_extended.antlr_parser import PSLexer, PSParser
except Exception:  # pragma: no cover
    PSLexer = None
    PSParser = None

from areal.reward import (
    _schedule_worker_suicide,
    _set_rlimit_as,
    _start_rss_killer_subprocess,
    _start_rss_watchdog,
)
from areal.utils import logging

logger = logging.getLogger("VerlMathReward")

_GOLD_TARGETS = (LatexExtractionConfig(),)
_PRED_TARGETS = (ExprExtractionConfig(), LatexExtractionConfig())
_TIMEOUT_S = 8.0  # tightened from 30s 2026-06-08 after FRPO retry-2 OOM:
# four reward workers reached 400 GB RSS each before Ray killed the node at
# step 86. RSS watchdog daemon thread failed to fire because long sympy
# parse held the GIL → daemon couldn't preempt. With 30s timeout an
# abandoned-thread leak bounds ~30s × ~80MB/s = 2.4 GB per pathological
# input; 8s caps at 640 MB. Trade-off: legitimately-long math problems
# (rare with 8K-tok cap) lose their verify and score 0.

# When the spawn-context reward worker's RSS exceeds this many MB, schedule its
# self-exit so a follower can replace it before the pathological math_verify
# call queues up another batch of 17 GB/4-min growth on this same process.
# Picked at ~6 GB because the steady-state worker holds ~2 GB of cached parser
# state; anything past 3× that is the runaway-leak signature observed on
# 2026-06-06 job 12610395 (5 GB → 87 GB over 16 min, killing the node).
# max_tasks_per_child=20 caught the slow per-call drift but did NOT prevent
# a single very-bad prompt from launching the worker straight into runaway,
# which is why FRPO retries-1/-2 OOMed at almost the same step (97/101).
_RSS_KILL_MB = 2500  # tightened 4 GB → 2.5 GB on 2026-06-08 after FRPO retry-2
# OOM: even with the 4 GB threshold the watchdog daemon couldn't fire fast
# enough — sympy parse held GIL → daemon starved → workers reached 400 GB
# per process. Earlier trigger + interval_s=0.5 gives the daemon more chances
# to catch the threshold during brief GIL release windows.
# Note: original constant was 4000 (per retry-1 working stack). Production OOMs grow workers to
                     # 400-500 GB at step ~80, suggesting individual workers
                     # cross the 6 GB threshold during a single timed-out
                     # ParserATN call. 4 GB gives more headroom for suicide
                     # to fire before the worker becomes catastrophic.

# Optional libc malloc_trim handle — resolved lazily so unit tests on macOS
# don't fail to import. Reduced glibc arena fragmentation by ~33% in repro
# (scripts/diag-notes.md H2). Called from the finally block per call.
_LIBC_MALLOC_TRIM = None


def _malloc_trim() -> None:
    global _LIBC_MALLOC_TRIM
    if _LIBC_MALLOC_TRIM is None:
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            _LIBC_MALLOC_TRIM = libc.malloc_trim
        except Exception:
            _LIBC_MALLOC_TRIM = False  # mark as tried-and-failed
    if _LIBC_MALLOC_TRIM and _LIBC_MALLOC_TRIM is not False:
        try:
            _LIBC_MALLOC_TRIM(0)
        except Exception:
            pass


_MAX_LATEX_DEPTH = 40  # parens/brackets — tighter than original 80 after
                       # 2026-06-07 GRPO retry-1 OOMed at step 86 despite
                       # the pre-check at 80; production inputs slip through.
_MAX_BACKSLASH_COUNT = 300  # tightened from 800 — same reason.
_MAX_RESPONSE_LEN_PRECHECK = 8192  # tightened from 16384.


# Per-worker counters so we can attribute reward=0 to its cause.
# Logged every 200 calls so the rate is visible without log-spamming.
_REJECT_COUNTERS = {
    "precheck_len": 0,
    "precheck_backslash": 0,
    "precheck_depth": 0,
    "math_exception": 0,
    "timeout": 0,
    "ok_nonzero": 0,
    "ok_zero": 0,
}


def _log_counters_if_due() -> None:
    total = sum(_REJECT_COUNTERS.values())
    if total > 0 and total % 200 == 0:
        try:
            logger.warning(f"[reward-attribution after {total} calls] {_REJECT_COUNTERS}")
        except Exception:
            pass


def _looks_pathological(response: str) -> bool:
    """Cheap pre-check: is this completion likely to push math_verify into
    a multi-second / multi-GB parse?

    The signature we want to catch: deeply-nested expressions emitted by
    late-training thinking traces. Synthetic measurements show depth=200
    paren nests cost 13 MB and 18 s; depth=500 costs 135 s. Production
    worker RSS would survive ~80-100 such inputs before Ray kills the node.
    Returning False here skips math_verify entirely and gives reward 0.
    A small fraction of correct answers may be missed, but unbounded RAM
    growth is the far worse outcome.

    Increments the appropriate _REJECT_COUNTERS entry on rejection so we
    can later attribute training-time reward collapse to its real cause.
    """
    if len(response) > _MAX_RESPONSE_LEN_PRECHECK:
        _REJECT_COUNTERS["precheck_len"] += 1
        return True
    bs = response.count("\\")
    if bs > _MAX_BACKSLASH_COUNT:
        _REJECT_COUNTERS["precheck_backslash"] += 1
        return True
    # Maximum simultaneous open-paren depth across the string
    depth = max_depth = 0
    for c in response:
        if c == "(" or c == "[" or c == "{":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif c == ")" or c == "]" or c == "}":
            depth -= 1
    if max_depth > _MAX_LATEX_DEPTH:
        _REJECT_COUNTERS["precheck_depth"] += 1
        return True
    return False


def _clear_antlr_dfa_cache() -> None:
    """Drop the class-level DFA states and shared context cache that
    latex2sympy2_extended's ANTLR4 parser accumulates per unique input.

    Belt-and-suspenders with _drop_math_verify_caches() above: even after
    we evict the sympy trees, the ANTLR prediction DFAs keep growing with
    input variety (see scripts/repro_leak.py investigation 2026-06-07).
    Clearing them costs nothing — the next parse rebuilds what it needs.
    """
    if PSParser is None or PSLexer is None:
        return
    try:
        for dfa in PSParser.decisionsToDFA:
            dfa.states.clear()
        for dfa in PSLexer.decisionsToDFA:
            dfa.states.clear()
        PSParser.sharedContextCache.cache.clear()
    except Exception:
        pass


def _maybe_schedule_rss_suicide() -> None:
    """If our own RSS is past _RSS_KILL_MB, schedule worker self-exit so the
    next reward call lands on a fresh process. Cheap: getrusage is a syscall."""
    try:
        # ru_maxrss is in kilobytes on Linux (POSIX leaves units implementation-
        # defined, but glibc uses KB here).
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return
    mb = kb / 1024.0
    if mb > _RSS_KILL_MB:
        logger.warning(
            f"Reward worker RSS={mb:.0f} MB exceeds {_RSS_KILL_MB} MB; "
            f"scheduling self-exit to recycle leaked C-level state"
        )
        _schedule_worker_suicide()


def _verify_impl(response: str, ground_truth_boxed: str) -> float:
    gold = parse(
        ground_truth_boxed, extraction_config=_GOLD_TARGETS, parsing_timeout=None
    )
    pred = parse(response, extraction_config=_PRED_TARGETS, parsing_timeout=None)
    if not gold or not pred:
        return 0.0
    return max(
        1.0
        if any(
            math_verify_verify(g, p, float_rounding=6, timeout_seconds=None)
            for g in gold
        )
        else 0.0
        for p in pred
    )


# See MathVerifyWorker._LONG_RESPONSE_CHARS for why we short-circuit long
# responses. verl_math is used for MATH500 eval, where the same degenerate
# long-CoT runaway shows up.
_LONG_RESPONSE_CHARS = 16384


def verl_math_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    # RSS-watchdog daemon — polls /proc/self/status every 5 s and calls
    # os._exit(137) if VmRSS > _RSS_KILL_MB. Graceful recycling: worker
    # exits between tasks rather than mid-task, so ProcessPoolExecutor
    # rebuilds clean (no BrokenProcessPool storm). FRPO retry-1 saw 187
    # train steps with just 3 pool breaks. GRPO retry-3 had RLIMIT_AS=12GB
    # instead → 211 BrokenProcessPool in 32 steps (kernel raised
    # MemoryError mid-task → catastrophic pool teardown). RLIMIT_AS
    # REMOVED — suicide alone is the working pattern.
    _start_rss_watchdog(threshold_mb=_RSS_KILL_MB)
    # GIL-independent backup: fork()s a killer subprocess on first call. The
    # in-process watchdog above starves under sympy's GIL-hogging C parses;
    # this one polls /proc and SIGKILLs the parent — kernel-level, no GIL.
    # Threshold is intentionally lower (2 GB vs 2.5 GB) so the killer fires
    # before the Python watchdog would have, catching the runaway earlier.
    _start_rss_killer_subprocess(threshold_mb=2000, interval_s=0.25)
    response = str(completions)
    if _looks_pathological(response):
        # See module docstring: pathological depth/length nests blow up sympy
        # parse to multi-GB single allocations and abandoned threads. Short-
        # circuit to 0 — losing a small fraction of valid rewards is cheaper
        # than worker-process OOM. (_REJECT_COUNTERS bumped inside.)
        _log_counters_if_due()
        return 0.0
    ground_truth_boxed = "\\boxed{" + str(answer) + "}"
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_verify_impl, response, ground_truth_boxed)
        result = future.result(timeout=_TIMEOUT_S)
        # Bucket the legitimate math_verify outcomes — distinguishes "model
        # answered correctly" from "model answered but wrong" from the
        # reject paths above. Together with precheck_* and math_exception
        # counters, this gives a complete attribution of every reward=0.
        if result > 0:
            _REJECT_COUNTERS["ok_nonzero"] += 1
        else:
            _REJECT_COUNTERS["ok_zero"] += 1
        _log_counters_if_due()
        return result
    except concurrent.futures.TimeoutError:
        logger.warning(
            f"Timeout ({_TIMEOUT_S}s) in verl_math_reward_fn for "
            f"ground_truth={ground_truth_boxed!r}; scheduling worker self-exit "
            f"so the GIL-hogging sympy thread cannot poison the next task"
        )
        _schedule_worker_suicide()
        _REJECT_COUNTERS["timeout"] += 1
        _log_counters_if_due()
        return 0.0
    except Exception:
        logger.warning("Exception in verl_math_reward_fn", exc_info=True)
        _REJECT_COUNTERS["math_exception"] += 1
        _log_counters_if_due()
        return 0.0
    finally:
        executor.shutdown(wait=False)
        _clear_antlr_dfa_cache()
        # malloc_trim before checking RSS so the threshold compares "real"
        # working set, not unreclaimed glibc arenas. Reduces fragmentation
        # by ~33% per scripts/diag-notes.md H2 finding.
        _malloc_trim()
        _maybe_schedule_rss_suicide()
