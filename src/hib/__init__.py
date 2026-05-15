"""Benchmark infrastructure for Hellinger imbalance experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hellinger-imbalance-benchmarks")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
