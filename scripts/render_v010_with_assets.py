from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

SRC = PAPER / "manuscript_v0_10.md"
OUT_MD = PAPER / "manuscript_v0_10_rendered.md"
OUT_HTML = PAPER / "manuscript_v0_10_rendered.html"
OUT_PDF = PAPER / "manuscript_v0_10_rendered.pdf"

FIGS = {
    "Figure 0": ("figure_0_operational_accessibility_lens.png", "Operational accessibility lens.", "95%"),
    "Figure 1": ("figure_1_reachability_mean_transition.png", "Mean reachability transition.", "80%"),
    "Figure 2": ("figure_2_reachability_by_dataset.png", "Dataset-level reachability panels.", "95%"),
    "Figure 3": ("figure_3_elasticity_interval_heatmap.png", "Elasticity interval concentration heatmap.", "80%"),
    "Figure 4": ("figure_4_support_vs_persistence.png", "Support vs persistence scatter.", "80%"),
    "Figure 5": ("figure_5_calibration_geometry_deltas.png", "Calibration geometry deltas.", "80%"),
    "Figure 7": ("figure_accessibility_regime_map_variant_a_v2.png", "Accessibility regime map: max jump vs threshold-survival persistence.", "80%"),
}


def strip_first_heading(md: str) -> str:
    return re.sub(r"^#\s+.*\n+", "", md.strip(), count=1)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    # Replace figures
    for label, (fname, cap, width) in FIGS.items():
        ph = f"[{label} about here]"
        img = f"![{cap}](manuscript_figures/{fname}){{width={width}}}\n"
        text = text.replace(ph, img)

    # Remove manual figure caption lines to avoid duplicates
    text = re.sub(r"\n\*\*Figure\s+\d+\:\*\*.*\n", "\n", text)

    # Tables (compact versions for readability)
    t1 = (PAPER / "rendered_review_draft" / "tables" / "table_1_compact.md").read_text(encoding="utf-8").strip()
    ta = (PAPER / "rendered_review_draft" / "tables" / "table_allocator_family_summary_compact.md").read_text(encoding="utf-8").strip()
    text = text.replace("[Table 1 about here]", t1)
    text = text.replace("[Table A about here]", ta)
    text = re.sub(r"\n\*\*Table\s+[^\n]*\n", "\n", text)

    OUT_MD.write_text(text, encoding="utf-8")

    subprocess.run(["pandoc", OUT_MD.name, "-s", "-o", OUT_HTML.name], check=True, cwd=str(PAPER))
    subprocess.run(["pandoc", OUT_MD.name, "-o", OUT_PDF.name, "--pdf-engine=typst"], check=True, cwd=str(PAPER))


if __name__ == "__main__":
    main()
