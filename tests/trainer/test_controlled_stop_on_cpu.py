import time

import pytest

from verl.trainer.controlled_stop import ControlledStopDataLoader


def test_controlled_stop_finishes_current_item_and_starts_no_new_item(monkeypatch):
    ticks = iter([0.0, 1.0, 7.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    loader = ControlledStopDataLoader([1, 2, 3], stop_after_seconds=5)
    assert list(loader) == [1]
    assert loader.controlled_stop_reached is True
    assert list(loader) == []


def test_absolute_deadline_includes_time_before_first_update(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(time, "time", lambda: 106.0)
    loader = ControlledStopDataLoader([1, 2], deadline_unix_seconds=105.0)
    assert list(loader) == []
    assert loader.controlled_stop_reached is True


def test_absolute_deadline_finishes_current_item(monkeypatch):
    wall_ticks = iter([100.0, 106.0])
    monkeypatch.setattr(time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(time, "time", lambda: next(wall_ticks))
    loader = ControlledStopDataLoader([1, 2], deadline_unix_seconds=105.0)
    assert list(loader) == [1]
    assert loader.controlled_stop_reached is True


def test_controlled_stop_rejects_two_deadline_modes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ControlledStopDataLoader([], stop_after_seconds=5, deadline_unix_seconds=100)


def test_controlled_stop_loader_requires_a_deadline():
    with pytest.raises(ValueError, match="must be configured"):
        ControlledStopDataLoader([])


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_controlled_stop_rejects_invalid_deadline(value):
    with pytest.raises(ValueError, match="finite positive"):
        ControlledStopDataLoader([], value)

    with pytest.raises(ValueError, match="finite positive"):
        ControlledStopDataLoader([], deadline_unix_seconds=value)
