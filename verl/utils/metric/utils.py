# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Metrics utils.
"""

from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch


def reduce_metrics(metrics: dict[str, Union["Metric", list[Any]]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if isinstance(val, Metric):
            metrics[key] = val.aggregate()
        elif "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


NumericType = int, float, torch.Tensor, np.ndarray
Numeric = int | float | torch.Tensor | np.ndarray


class Metric:
    """
    A metric aggregator for collecting and aggregating numeric values.

    This class accumulates numeric values (int, float, or scalar tensors) and computes
    an aggregate statistic based on the specified aggregation type (MEAN, SUM, MIN, or MAX).

    Args:
        aggregation: The aggregation method to use. Can be a string ("mean", "sum", "min", "max")
            or an AggregationType enum value.
        value: Optional initial value(s) to add. Can be a single numeric value or a list of values.

    Example:
        >>> metric = Metric(aggregation="mean", value=1.0)
        >>> metric.append(2.0)
        >>> metric.append(3.0)
        >>> metric.aggregate()
        2.0
    """

    def __init__(
        self,
        aggregation: str | AggregationType,
        value: Optional[Numeric | list[Numeric]] = None,
        weight: Numeric = 1.0,
    ) -> None:
        if isinstance(aggregation, str):
            self.aggregation = AggregationType(aggregation)
        else:
            self.aggregation = aggregation
        if not isinstance(self.aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        self.values = []
        self.weights = []
        if value is not None:
            self.append(value, weight=weight)

    @staticmethod
    def _coerce_weight(weight: Numeric) -> float:
        if isinstance(weight, torch.Tensor):
            if weight.numel() != 1:
                raise ValueError("Only scalar metric weights are supported")
            weight = weight.detach().item()
        if not isinstance(weight, NumericType):
            raise ValueError(f"Unsupported metric weight type: {type(weight)}")
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"Metric weight must be finite and non-negative, got {weight}.")
        return weight

    def append(self, value: Union[Numeric, "Metric"], weight: Numeric = 1.0) -> None:
        if isinstance(value, Metric):
            if self._coerce_weight(weight) != 1.0:
                raise ValueError("An appended Metric already owns its weights; an additional weight is not allowed.")
            self.extend(value)
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be converted to float")
            value = value.detach().item()
        if not isinstance(value, NumericType):
            raise ValueError(f"Unsupported value type: {type(value)}")
        self.values.append(value)
        self.weights.append(self._coerce_weight(weight))

    def extend(self, values: Union["Metric", list[Numeric]]) -> None:
        if isinstance(values, Metric):
            if values.aggregation != self.aggregation:
                raise ValueError(f"Aggregation type mismatch: {self.aggregation} != {values.aggregation}")
            for value, weight in zip(values.values, values.weights, strict=True):
                self.append(value, weight=weight)
            return
        for value in values:
            self.append(value)

    def aggregate(self) -> float:
        return self._aggregate(self.values, self.aggregation, self.weights)

    @classmethod
    def _aggregate(
        cls,
        values: list[Numeric],
        aggregation: AggregationType,
        weights: Optional[list[Numeric]] = None,
    ) -> float:
        if weights is None:
            weights = [1.0] * len(values)
        if len(values) != len(weights):
            raise ValueError("Metric values and weights must have identical lengths.")
        value_array = np.asarray(values)
        weight_array = np.asarray(weights, dtype=np.float64)
        active = weight_array > 0
        if not active.any():
            if aggregation == AggregationType.SUM:
                return 0.0
            raise ValueError("Cannot aggregate a metric with zero total scientific weight.")
        match aggregation:
            case AggregationType.MEAN:
                return np.average(value_array, weights=weight_array)
            case AggregationType.SUM:
                return np.sum(value_array[active])
            case AggregationType.MIN:
                return np.min(value_array[active])
            case AggregationType.MAX:
                return np.max(value_array[active])

    @classmethod
    def aggregate_dp(cls, metric_lists: list["Metric"]) -> float:
        if not metric_lists:
            raise ValueError("Cannot aggregate an empty list of metrics.")
        value_lists = [ml.values for ml in metric_lists]
        weight_lists = [ml.weights for ml in metric_lists]
        for rank, (values, weights) in enumerate(zip(value_lists, weight_lists, strict=True)):
            if len(values) != len(weights):
                raise ValueError(
                    "Metric values and weights must have identical lengths "
                    f"on dp rank {rank}: {len(values)} != {len(weights)}."
                )
        if not all(ml.aggregation == metric_lists[0].aggregation for ml in metric_lists):
            raise ValueError("All Metric instances must use the same aggregation for dp aggregation.")
        aggregation = metric_lists[0].aggregation
        match aggregation:
            case AggregationType.SUM:
                # Loss values are already scaled by dp_size. Preserve the
                # historical mean-over-DP, sum-over-accumulation contract while
                # allowing ranks to use different numbers of dynamically packed
                # microbatches. Explicitly inactive padding still contributes 0.
                rank_sums = [
                    cls._aggregate(values, aggregation, weights)
                    for values, weights in zip(value_lists, weight_lists)
                ]
                return np.mean(rank_sums)
            case AggregationType.MEAN | AggregationType.MIN | AggregationType.MAX:
                flat_values = [value for values in value_lists for value in values]
                flat_weights = [weight for weights in weight_lists for weight in weights]
                return cls._aggregate(values=flat_values, aggregation=aggregation, weights=flat_weights)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Numeric],
        aggregation: str | AggregationType,
        weight: Numeric = 1.0,
    ) -> dict[str, "Metric"]:
        return {key: cls(value=value, aggregation=aggregation, weight=weight) for key, value in data.items()}

    def init_list(self) -> "Metric":
        return Metric(aggregation=self.aggregation)
