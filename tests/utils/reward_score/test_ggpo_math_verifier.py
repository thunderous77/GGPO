import pytest

from examples.ggpo import math_verifier


def test_math_verifier_returns_binary_dict():
    correct = math_verifier.compute_score(r"The answer is \boxed{2}.", "2", timeout=10.0)
    wrong = math_verifier.compute_score(r"The answer is \boxed{3}.", "2", timeout=10.0)

    assert correct == {
        "score": 1.0,
        "acc": 1.0,
        "invalid": 0.0,
        "verifier_timeout": 0.0,
        "parser_success": 1.0,
    }
    assert wrong == {
        "score": 0.0,
        "acc": 0.0,
        "invalid": 0.0,
        "verifier_timeout": 0.0,
        "parser_success": 1.0,
    }


def test_math_verifier_scores_unboxed_answers_directly():
    result = math_verifier.compute_score("The answer is 2.", "2")

    assert result == {
        "score": 1.0,
        "acc": 1.0,
        "invalid": 0.0,
        "verifier_timeout": 0.0,
        "parser_success": 1.0,
    }
    assert math_verifier.compute_score("The answer is 3.", "2")["score"] == 0.0


def test_math_verifier_timeout_sets_timeout_flag(monkeypatch):
    def timeout(*_):
        raise TimeoutError

    monkeypatch.setattr(math_verifier.math_verify, "_verify_in_subprocess", timeout)

    result = math_verifier.compute_score(r"\boxed{2}", "2", timeout=0.01)

    assert result == {
        "score": 0.0,
        "acc": 0.0,
        "invalid": 0.0,
        "verifier_timeout": 1.0,
        "parser_success": 1.0,
    }


def test_math_verifier_does_not_create_nested_pool(monkeypatch):
    monkeypatch.setattr(math_verifier.math_verify, "_verify_in_subprocess", lambda *_: 1.0)
    monkeypatch.setattr(
        math_verifier.math_verify,
        "_get_pool",
        lambda: (_ for _ in ()).throw(AssertionError("must not create a nested process pool")),
    )

    result = math_verifier.compute_score(r"\boxed{2}", "2")

    assert result == {
        "score": 1.0,
        "acc": 1.0,
        "invalid": 0.0,
        "verifier_timeout": 0.0,
        "parser_success": 1.0,
    }


@pytest.mark.parametrize("solution", [None, 123, {"boxed": "2"}])
def test_math_verifier_non_string_solution_is_invalid(solution):
    result = math_verifier.compute_score(solution, "2")

    assert result == {
        "score": 0.0,
        "acc": 0.0,
        "invalid": 1.0,
        "verifier_timeout": 0.0,
        "parser_success": 0.0,
    }


def test_math_verifier_exception_sets_invalid_flag(monkeypatch):
    def fail(*_):
        raise RuntimeError("verifier failed")

    monkeypatch.setattr(math_verifier.math_verify, "_verify_in_subprocess", fail)

    result = math_verifier.compute_score(r"\boxed{2}", "2")

    assert result == {
        "score": 0.0,
        "acc": 0.0,
        "invalid": 1.0,
        "verifier_timeout": 0.0,
        "parser_success": 1.0,
    }


def test_math_verifier_rejects_fractional_helper_scores(monkeypatch):
    monkeypatch.setattr(math_verifier.math_verify, "_verify_in_subprocess", lambda *_: 0.25)

    result = math_verifier.compute_score(r"\boxed{2}", "2")

    assert result["score"] == 0.0
    assert result["acc"] == 0.0
    assert result["score"] in (0.0, 1.0)
    assert result["acc"] in (0.0, 1.0)


@pytest.mark.parametrize(
    ("helper_score", "expected_score"),
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (True, 1.0),
        (False, 0.0),
    ],
)
def test_math_verifier_normalizes_boolean_helper_scores(monkeypatch, helper_score, expected_score):
    def verify(ground_truth, solution):
        assert ground_truth == r"\boxed{2}"
        assert solution == r"\boxed{2}"
        return helper_score

    monkeypatch.setattr(math_verifier.math_verify, "_verify_in_subprocess", verify)

    result = math_verifier.compute_score(r"\boxed{2}", "2")

    assert result == {
        "score": expected_score,
        "acc": expected_score,
        "invalid": 0.0,
        "verifier_timeout": 0.0,
        "parser_success": 1.0,
    }
