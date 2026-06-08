from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_vultr_script_contains_expected_steps():
    script = ROOT / "scripts" / "bootstrap_vultr.sh"
    text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert "apt update" in text
    assert "git curl build-essential python3 python3-venv htop tmux rsync" in text
    assert "https://astral.sh/uv/install.sh" in text
    assert "REPO_URL" in text
    assert "research/accessibility-next-experiments" in text
    assert "uv sync --dev" in text
    assert "uv pip install -e" in text
    assert "pytest" in text


def test_run_vultr_density_threshold_script_contains_expected_steps():
    script = ROOT / "scripts" / "run_vultr_density_threshold.sh"
    text = script.read_text(encoding="utf-8")

    assert script.exists()
    assert "source .venv/bin/activate" in text
    assert "export PYTHONPATH=src" in text
    assert "--preset density_threshold" in text
    assert "mlp_weighted_bce,mlp_weighted_bce_dropout_0_1" in text
    assert "SEED_SHARDS" in text
    assert "density_threshold_sweep.csv" in text
    assert "density_threshold_sweep_summary.md" in text
    assert "EXPECTED_FULL_CSV_LINES=2241" in text
    assert "scp" in text
    assert "rsync" in text
