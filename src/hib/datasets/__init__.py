"""Dataset ingestion and registry utilities."""

from .registry import LEGACY_HDDT_DATASET_REGISTRY
from .legacy_registry import LEGACY_HDDT_DATASET_REGISTRY as CURATED_LEGACY_HDDT_DATASET_REGISTRY

__all__ = [
    "LEGACY_HDDT_DATASET_REGISTRY",
    "CURATED_LEGACY_HDDT_DATASET_REGISTRY",
]
