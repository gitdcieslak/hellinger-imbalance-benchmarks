"""Visualization helpers for prediction-space occupancy artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from hib.scores import HISTOGRAM_BINS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _as_points(items: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["x"] for row in items], dtype=float)
    y = np.asarray([row["y"] for row in items], dtype=float)
    return x, y


def plot_occupancy_artifacts(records: list[dict], output_dir: str | Path) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for rec in records:
        dataset_id = str(rec.get("dataset_id", rec.get("skew_ratio", "unknown")))
        model_id = str(rec["model_id"])
        split_id = str(rec.get("split_id", "0"))
        split_slug = split_id.replace(" ", "_")
        prefix = f"{dataset_id}_{model_id}_{split_slug}"
        metrics = rec["metrics"]

        # Histogram
        labels = list(metrics["histogram_counts"]["all"].keys())
        pos = np.asarray([metrics["histogram_counts"]["positive"][k] for k in labels], dtype=float)
        neg = np.asarray([metrics["histogram_counts"]["negative"][k] for k in labels], dtype=float)
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - 0.2, neg, width=0.4, label="negative", alpha=0.7)
        ax.bar(x + 0.2, pos, width=0.4, label="positive", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"Class-Conditional Score Histograms: {dataset_id} {model_id}")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        p1 = out_dir / f"{prefix}_histogram.png"
        s1 = out_dir / f"{prefix}_histogram.svg"
        fig.savefig(p1, dpi=120)
        fig.savefig(s1)
        plt.close(fig)
        created.extend([p1, s1])

        # ECDF
        px, py = _as_points(metrics["ecdf"]["positive"])
        nx, ny = _as_points(metrics["ecdf"]["negative"])
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(nx, ny, label="negative")
        ax2.plot(px, py, label="positive")
        for thr in [0.50, 0.25, 0.10, 0.05, 0.01]:
            ax2.axvline(thr, color="gray", linestyle="--", alpha=0.4)
        ax2.set_xlim(0.0, 1.0)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_xlabel("Score")
        ax2.set_ylabel("ECDF")
        ax2.set_title(f"ECDF Occupancy: {dataset_id} {model_id}")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")
        fig2.tight_layout()
        p2 = out_dir / f"{prefix}_ecdf.png"
        s2 = out_dir / f"{prefix}_ecdf.svg"
        fig2.savefig(p2, dpi=120)
        fig2.savefig(s2)
        plt.close(fig2)
        created.extend([p2, s2])

        # Threshold occupancy trajectory
        traj = metrics["threshold_occupancy"]
        th = np.asarray([row["threshold"] for row in traj], dtype=float)
        po = np.asarray([row["positive_occupancy"] for row in traj], dtype=float)
        no = np.asarray([row["negative_occupancy"] for row in traj], dtype=float)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(th, po, marker="o", label="positive occupancy")
        ax3.plot(th, no, marker="o", label="negative occupancy")
        ax3.set_xlim(0.5, 0.01)
        ax3.set_ylim(0.0, 1.0)
        ax3.set_xlabel("Threshold")
        ax3.set_ylabel("Occupancy fraction")
        ax3.set_title(f"Threshold Occupancy Trajectory: {dataset_id} {model_id}")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best")
        fig3.tight_layout()
        p3 = out_dir / f"{prefix}_threshold_occupancy.png"
        s3 = out_dir / f"{prefix}_threshold_occupancy.svg"
        fig3.savefig(p3, dpi=120)
        fig3.savefig(s3)
        plt.close(fig3)
        created.extend([p3, s3])

        # Quantization score panel
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.bar(["quantization"], [float(metrics["quantization_score"])])
        ax4.set_ylim(0.0, 1.0)
        ax4.set_title(f"Quantization Score: {dataset_id} {model_id}")
        ax4.grid(True, axis="y", alpha=0.3)
        fig4.tight_layout()
        p4 = out_dir / f"{prefix}_quantization.png"
        s4 = out_dir / f"{prefix}_quantization.svg"
        fig4.savefig(p4, dpi=120)
        fig4.savefig(s4)
        plt.close(fig4)
        created.extend([p4, s4])

    return created
