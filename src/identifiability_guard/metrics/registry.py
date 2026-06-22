"""Metric registry for unified API across all metrics."""

from typing import Dict, Type, Optional, List
import numpy as np
import warnings

from .base import BaseMetric, MetricResult


class MetricRegistry:
    """
    Unified API for all identifiability metrics.

    Allows registration and creation of metrics by name, providing a consistent
    interface regardless of implementation source (internal, DisentanglementLib, etc.).

    Example:
        >>> from src.metrics import MetricRegistry
        >>> registry = MetricRegistry()
        >>>
        >>> # Register metrics
        >>> registry.register_defaults()
        >>>
        >>> # Create and use a metric
        >>> metric = registry.create("dci")
        >>> result = metric.compute(Z, Z_hat)
        >>> print(f"Disentanglement: {result.subscores['disentanglement']:.3f}")
        >>>
        >>> # Compute all metrics at once
        >>> results = registry.compute_all(Z, Z_hat)
        >>> for name, result in results.items():
        >>>     print(f"{name}: {result.primary_score:.3f}")
    """

    def __init__(self):
        """Initialize empty registry."""
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        self._default_kwargs: Dict[str, dict] = {}
        self._display_names: Dict[str, str] = {}

    def register(
        self,
        name: str,
        metric_cls: Type[BaseMetric],
        default_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Register a metric class.

        Args:
            name: Unique name for the metric (e.g., "dci", "mcc_pearson").
            metric_cls: The metric class (must inherit from BaseMetric).
            default_kwargs: Default keyword arguments for instantiation.

        Raises:
            ValueError: If name is already registered or metric_cls is invalid.
        """
        normalized = name.lower()
        if normalized in self._metrics:
            raise ValueError(
                f"Metric '{name}' is already registered. "
                f"Use unregister('{name}') first."
            )

        if not issubclass(metric_cls, BaseMetric):
            raise ValueError(
                f"metric_cls must inherit from BaseMetric, "
                f"got {metric_cls.__name__}"
            )

        # store a copy to avoid external mutation of defaults
        self._metrics[normalized] = metric_cls
        self._default_kwargs[normalized] = dict(default_kwargs or {})
        self._display_names[normalized] = name

    def unregister(self, name: str) -> None:
        """
        Unregister a metric.

        Args:
            name: Name of the metric to unregister.

        Raises:
            ValueError: If metric is not registered.
        """
        normalized = name.lower()
        if normalized not in self._metrics:
            raise ValueError(
                f"Metric '{name}' is not registered. "
                f"Available: {self.list_metrics()}"
            )

        del self._metrics[normalized]
        del self._default_kwargs[normalized]
        del self._display_names[normalized]

    def create(self, name: str, **kwargs) -> BaseMetric:
        """
        Instantiate a metric by name.

        Args:
            name: Name of the registered metric.
            **kwargs: Keyword arguments to override defaults.

        Returns:
            Instantiated metric.

        Raises:
            ValueError: If metric is not registered.

        Example:
            >>> metric = registry.create("dci", train_test_split=0.7)
        """
        normalized = name.lower()
        if normalized not in self._metrics:
            raise ValueError(
                f"Unknown metric: '{name}'. Available: {self.list_metrics()}"
            )

        # Merge default kwargs with provided kwargs
        merged_kwargs = {**self._default_kwargs[normalized], **kwargs}

        return self._metrics[normalized](**merged_kwargs)

    def list_metrics(self) -> List[str]:
        """Return list of registered metric names."""
        return sorted(self._display_names.values())

    def compute_all(
        self,
        Z: np.ndarray,
        Z_hat: np.ndarray,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, MetricResult]:
        """
        Compute all registered metrics (or a subset).

        Args:
            Z: Ground-truth factors of shape (n, d).
            Z_hat: Learned codes of shape (n, m).
            metric_names: Optional list of metric names to compute.
                        If None, computes all registered metrics.

        Returns:
            Dictionary mapping metric names to MetricResult objects.

        Example:
            >>> results = registry.compute_all(Z, Z_hat)
            >>> results = registry.compute_all(Z, Z_hat, metric_names=["dci", "mcc"])
        """
        names = metric_names if metric_names is not None else self.list_metrics()

        results = {}
        for name in names:
            metric = self.create(name)
            try:
                results[name] = metric.compute(Z, Z_hat)
            except Exception as e:
                # Continue with other metrics if one fails
                warnings.warn(f"{name} failed with error: {e}")
                continue

        return results

    def compute_all_oos(
        self,
        Z_train: np.ndarray,
        Z_hat_train: np.ndarray,
        Z_test: np.ndarray,
        Z_hat_test: np.ndarray,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, MetricResult]:
        """
        Compute all registered metrics with out-of-sample evaluation.

        Metrics that override ``compute_oos`` (e.g. R², InfoE) will fit on the
        training split and evaluate on the test split.  All other metrics fall
        back to evaluating on the test split only (the ``BaseMetric`` default).

        Args:
            Z_train: Ground-truth factors for fitting, shape (n_train, d).
            Z_hat_train: Learned codes for fitting, shape (n_train, m).
            Z_test: Ground-truth factors for evaluation, shape (n_test, d).
            Z_hat_test: Learned codes for evaluation, shape (n_test, m).
            metric_names: Optional list of metric names to compute.
                        If None, computes all registered metrics.

        Returns:
            Dictionary mapping metric names to MetricResult objects.
        """
        names = metric_names if metric_names is not None else self.list_metrics()

        results = {}
        for name in names:
            metric = self.create(name)
            try:
                results[name] = metric.compute_oos(
                    Z_train, Z_hat_train, Z_test, Z_hat_test,
                )
            except Exception as e:
                warnings.warn(f"{name} failed with error: {e}")
                continue

        return results

    def register_defaults(self) -> None:
        """
        Register the default set of metrics.

        Registers:
        - dci: Disentanglement, Completeness, Informativeness
        - mcc_pearson: Mean Correlation Coefficient (Pearson)
        - mcc_spearman: Mean Correlation Coefficient (Spearman)
        - mcc_rdc: Mean Correlation Coefficient (RDC)
        - r2: R² score
        - mig: Mutual Information Gap
        - tmex: Testing for Measurement Exchangeability
        - infom: Information-theoretic Modularity
        - infoe: Information-theoretic Explicitness
        - infoc: Information-theoretic Compactness
        """
        from .dci import DCIMetric
        from .mcc import MCCMetric
        from .r2 import R2Metric
        from .mig import MIGMetric
        from .tmex import TMEXMetric
        from .infomec import InfoMMetric, InfoEMetric, InfoCMetric

        self.register("dci", DCIMetric, default_kwargs={"train_test_split": 0.8})
        self.register(
            "mcc_pearson", MCCMetric, default_kwargs={"method": "pearson"}
        )
        self.register(
            "mcc_spearman", MCCMetric, default_kwargs={"method": "spearman"}
        )
        self.register("mcc_rdc", MCCMetric, default_kwargs={"method": "rdc"})
        self.register("r2", R2Metric)
        self.register("mig", MIGMetric, default_kwargs={"num_bins": 20})
        self.register(
            "tmex", TMEXMetric, 
            default_kwargs={"alpha": 0.05, "regression_method": "lm", "rep": 9}
        )
        self.register("infom", InfoMMetric, default_kwargs={"discrete_latents": False})
        self.register("infoe", InfoEMetric, default_kwargs={"discrete_latents": False})
        self.register("infoc", InfoCMetric, default_kwargs={"discrete_latents": False})

    def __repr__(self) -> str:
        """String representation showing registered metrics."""
        metrics = self.list_metrics()
        return f"MetricRegistry({len(metrics)} metrics: {metrics})"
