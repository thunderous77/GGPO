from verl.utils import reward_score
from verl.utils.reward_score import math_verify


def test_get_default_compute_score_returns_batch_scorer_for_batch_manager():
    assert reward_score.get_default_compute_score("batch") is reward_score.batch_compute_score
    assert reward_score.get_default_compute_score("naive") is reward_score.default_compute_score


def test_batch_compute_score_scores_boxed_alp_math_outputs_without_process_pool(monkeypatch):
    calls = []

    def fake_parallel_compute_score(*args, **kwargs):
        raise AssertionError("batch ALP math scoring should avoid nested verifier process pools")

    def fake_compute_boxed_score(model_output, ground_truth, timeout_score):
        calls.append((model_output, ground_truth, timeout_score))
        return 1.0 if ground_truth == "2" else 0.0

    monkeypatch.setattr(math_verify, "parallel_compute_score", fake_parallel_compute_score)
    monkeypatch.setattr(math_verify, "_compute_boxed_score", fake_compute_boxed_score)

    scores = reward_score.batch_compute_score(
        data_sources=["alp_guru_rl92k_math", "alp_openr1_filtered", "alp_math500", "alp_guru_rl92k_math"],
        solution_strs=["no final answer", r"\boxed{2}", r"\boxed{}", r"\boxed{3}"],
        ground_truths=["2", "2", "2", "4"],
        timeout=7,
        num_processes=3,
        timeout_score=-1.0,
    )

    assert scores == [-1.0, 1.0, -1.0, 0.0]
    assert calls == [(r"\boxed{2}", "2", -1.0), (r"\boxed{3}", "4", -1.0)]


def test_compute_boxed_score_treats_math_verify_base_exception_timeout_as_zero(monkeypatch):
    class BaseExceptionTimeout(math_verify.TimeoutException):
        pass

    def timeout_verifier(*_args, **_kwargs):
        raise BaseExceptionTimeout("verification timed out")

    monkeypatch.setattr(math_verify, "_ensure_batch_verify_func", lambda: timeout_verifier)

    assert math_verify._compute_boxed_score(r"\boxed{2}", "2", timeout_score=0.0) == 0.0


def test_batch_compute_score_group_debug_is_opt_in_and_tolerates_invalid_sample_limit(monkeypatch, capsys):
    monkeypatch.setenv("VERL_GROUP_REWARD_DEBUG", "1")
    monkeypatch.setenv("VERL_GROUP_REWARD_DEBUG_GROUP_SIZE", "4")
    monkeypatch.setenv("VERL_GROUP_REWARD_DEBUG_SAMPLE_GROUPS", "invalid")
    monkeypatch.setattr(math_verify, "extract_boxed_answer", lambda _: True)
    monkeypatch.setattr(
        math_verify,
        "_compute_boxed_score",
        lambda model_output, ground_truth, timeout_score: float(ground_truth),
    )

    scores = reward_score.batch_compute_score(
        data_sources=["alp_math500"] * 8,
        solution_strs=[r"\boxed{0}"] * 8,
        ground_truths=["0", "0", "0", "0", "1", "0", "1", "0"],
    )

    assert scores == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert "[batch_group_rewards] group_size=4 groups=2 all_zero=1 mixed=1 all_one=0 " in capsys.readouterr().out
