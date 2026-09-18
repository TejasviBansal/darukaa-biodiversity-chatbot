"""Classify raw metric values into sourced scientific threshold bands."""

import logging
import math

from config import CATEGORICAL_METRICS, CATEGORICAL_SOURCES, THRESHOLDS, VALID_RANGES

logger = logging.getLogger(__name__)


def classify_metric(metric_name: str, value: float | str) -> dict[str, str | float]:
    """
    Classify one metric value into its threshold band or known category.

    Returns a dict: {"metric_name", "value", "band_label", "source"}.
    Raises ValueError if the metric is unknown, the value is invalid
    (non-finite, outside the physically valid range, or unknown), or no band matches.
    """
    if metric_name in CATEGORICAL_METRICS:
        if not isinstance(value, str):
            raise ValueError(f"Value for {metric_name} must be a category string: {value}")
        if value not in CATEGORICAL_METRICS[metric_name]:
            known_values = ", ".join(CATEGORICAL_METRICS[metric_name])
            raise ValueError(f"Value for {metric_name} must be one of: {known_values}")
        return {
            "metric_name": metric_name,
            "value": value,
            "band_label": value,
            "source": CATEGORICAL_SOURCES[metric_name],
        }

    if metric_name not in THRESHOLDS:
        raise ValueError(f"Unknown metric: {metric_name}")

    if metric_name not in VALID_RANGES:
        raise ValueError(f"Missing valid range for metric: {metric_name}")

    if not isinstance(value, int | float):
        raise ValueError(f"Value for {metric_name} must be numeric: {value}")

    if not math.isfinite(value):
        raise ValueError(f"Invalid non-finite value for {metric_name}: {value}")

    min_valid, max_valid = VALID_RANGES[metric_name]
    if value < min_valid or value > max_valid:
        raise ValueError(
            f"Value for {metric_name} must be between {min_valid} and {max_valid}: {value}",
        )

    bands = THRESHOLDS[metric_name]
    for index, band in enumerate(bands):
        is_last_band = index == len(bands) - 1
        if is_last_band:
            in_band = band.min_value <= value <= band.max_value
        else:
            in_band = band.min_value <= value < band.max_value
        if in_band:
            return {
                "metric_name": metric_name,
                "value": value,
                "band_label": band.label,
                "source": band.source,
            }

    raise ValueError(f"No threshold band matched {metric_name}={value}")


def classify_all(metrics: dict[str, float | str]) -> dict[str, dict[str, str | float]]:
    """
    Classify every metric in the input dict that has configured thresholds.

    Unknown metric names are logged as a warning and skipped, not raised.
    Returns {metric_name: classification_dict}.
    """
    classifications: dict[str, dict[str, str | float]] = {}
    for metric_name, value in metrics.items():
        if metric_name not in THRESHOLDS and metric_name not in CATEGORICAL_METRICS:
            logger.warning("Skipping unknown metric: %s", metric_name)
            continue
        classifications[metric_name] = classify_metric(metric_name, value)
    return classifications
