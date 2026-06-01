from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "rendered_review_draft"
TABLE_OUT = OUT / "tables"

SRC = PAPER / "manuscript_v0_9a.md"
RENDERED_MD = OUT / "manuscript_submission_draft.md"
RENDERED_HTML = OUT / "manuscript_submission_draft.html"
RENDERED_PDF = OUT / "manuscript_submission_draft.pdf"
REPORT = OUT / "submission_render_report.md"
APP_NOTES = OUT / "appendix_source_notes.md"

FIGURES = {
    "Figure 0": ("figure_0_operational_accessibility_lens.png", "Operational accessibility lens.", "95%"),
    "Figure 1": ("figure_1_reachability_mean_transition.png", "Mean reachability transition.", "80%"),
    "Figure 2": ("figure_2_reachability_by_dataset.png", "Dataset-level reachability panels.", "95%"),
    "Figure 3": ("figure_3_elasticity_interval_heatmap.png", "Elasticity interval concentration heatmap.", "80%"),
    "Figure 4": ("figure_4_support_vs_persistence.png", "Support vs persistence scatter.", "80%"),
    "Figure 5": ("figure_5_calibration_geometry_deltas.png", "Calibration geometry deltas.", "80%"),
    "Figure 7": (
        "figure_accessibility_regime_map_variant_a_v2.png",
        "Accessibility regime map: max jump vs threshold-survival persistence.",
        "80%",
    ),
}

CITE_MAP = {
    "Elkan 2001": "Elkan2001",
    "Chawla et al. 2002": "Chawla2002",
    "He and Garcia 2009": "HeGarcia2009",
    "Krawczyk 2016": "Krawczyk2016",
    "Buda et al. 2018": "Buda2018",
    "Johnson and Khoshgoftaar 2019": "JohnsonKhoshgoftaar2019",
    "Cieslak and Chawla 2008": "CieslakChawla2008",
    "Cieslak et al. 2012": "CieslakEtAl2012",
    "Davis and Goadrich 2006": "DavisGoadrich2006",
    "Saito and Rehmsmeier 2015": "SaitoRehmsmeier2015",
    "Fawcett 2006": "Fawcett2006",
    "Drummond and Holte 2006": "DrummondHolte2006",
    "Chow 1970": "Chow1970",
    "El-Yaniv and Wiener 2010": "ElYanivWiener2010",
    "Geifman and El-Yaniv 2017": "GeifmanElYaniv2017",
    "Platt 1999": "Platt1999",
    "Zadrozny and Elkan 2002": "ZadroznyElkan2002",
    "Niculescu-Mizil and Caruana 2005": "NiculescuMizilCaruana2005",
    "Guo et al. 2017": "Guo2017",
    "Sculley et al. 2015": "Sculley2015",
    "Breck et al. 2017": "Breck2017",
    "Amershi et al. 2019": "Amershi2019",
}


def parse_markdown_table(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = [ln.rstrip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.startswith("|")]
    header = [c.strip() for c in table_lines[0].strip("|").split("|")]
    rows = []
    for ln in table_lines[2:]:
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    return header, rows


def compact_tables() -> tuple[Path, Path]:
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    t1_src = PAPER / "manuscript_tables" / "table_1_ranking_vs_accessibility.md"
    ta_src = PAPER / "manuscript_tables" / "table_allocator_family_summary_v2.md"
    _, rows1 = parse_markdown_table(t1_src)
    _, rowsa = parse_markdown_table(ta_src)

    t1 = TABLE_OUT / "table_1_compact.md"
    t1_lines = [
        "| Model | AUROC | AP | R@0.50 | R@0.01 | Recovery | Smooth. | Max Jump | Pattern |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows1:
        t1_lines.append(
            f"| {r[0]} | {float(r[1]):.3f} | {float(r[2]):.3f} | {float(r[3]):.3f} | {float(r[4]):.3f} | {float(r[5]):.3f} | {float(r[6]):.3f} | {float(r[7]):.3f} | {r[8]} |"
        )
    t1.write_text("\n".join(t1_lines) + "\n", encoding="utf-8")

    ta = TABLE_OUT / "table_allocator_family_summary_compact.md"
    ta_lines = [
        "| Model | AUROC | AP | R@0.50 | R@0.01 | Recovery | Persistence | Max Jump | Pattern |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rowsa:
        ta_lines.append(
            f"| {r[0]} | {float(r[1]):.3f} | {float(r[2]):.3f} | {float(r[3]):.3f} | {float(r[4]):.3f} | {float(r[5]):.3f} | {float(r[6]):.3f} | {float(r[8]):.3f} | {r[9]} |"
        )
    ta.write_text("\n".join(ta_lines) + "\n", encoding="utf-8")
    return t1, ta


def convert_citations(text: str) -> tuple[str, int, list[str]]:
    unresolved: list[str] = []
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        label = m.group(1).strip()
        key = CITE_MAP.get(label)
        if key:
            count += 1
            return f"[@{key}]"
        unresolved.append(label)
        return m.group(0)

    text = re.sub(r"\[CITATION:\s*([^\]]+)\]", repl, text)
    return text, count, sorted(set(unresolved))


def build_manuscript(t1: Path, ta: Path) -> tuple[list[str], list[str], int, int, int, int]:
    text = SRC.read_text(encoding="utf-8")
    found_fig: list[str] = []
    missing_fig: list[str] = []

    # Equation block conversion
    text = text.replace("\\[\nR(t) = P(\\hat{p}(x) \\ge t \\mid y=1)\n\\]", "$$\nR(t) = P(\\hat{p}(x) \\ge t \\mid y=1)\n$$")

    # Figure placeholders -> images
    for label, (fname, caption, width) in FIGURES.items():
        ph = f"[{label} about here]"
        fpath = PAPER / "manuscript_figures" / fname
        if fpath.exists():
            img = f"![{caption}](../manuscript_figures/{fname}){{width={width}}}\n"
            text = text.replace(ph, img)
            found_fig.append(fname)
        else:
            missing_fig.append(fname)

    # Remove duplicate manual figure captions
    dup_removed = len(re.findall(r"\*\*Figure\s+\d+\:\*\*.*", text))
    text = re.sub(r"\n\*\*Figure\s+\d+\:\*\*.*\n", "\n", text)

    # Replace table placeholders with compact tables
    text = text.replace("[Table 1 about here]", t1.read_text(encoding="utf-8").strip())
    text = text.replace("[Table A about here]", ta.read_text(encoding="utf-8").strip())

    # Remove manual table captions
    text = re.sub(r"\n\*\*Table\s+[^\n]*\n", "\n", text)

    # Add short note where table captions were
    text = text.replace(
        "To establish that accessibility morphology appears before neural perturbation analysis, we next summarize family-level operational behavior under the same protocol.",
        "To establish that accessibility morphology appears before neural perturbation analysis, we next summarize family-level operational behavior under the same protocol. Values are dataset means over the severe-imbalance slice; see Appendix Source Notes.",
    )

    # Citation conversion
    text, converted_count, unresolved = convert_citations(text)

    # Append source notes appendix reference
    text += "\n\n## Appendix Source Notes\n\nSee `appendix_source_notes.md` in this review-draft package for table sources and definitions.\n"

    unresolved_about_here = len(re.findall(r"about here", text))
    unresolved_citation_tokens = len(re.findall(r"\[CITATION:\s*[^\]]+\]", text))

    RENDERED_MD.write_text(text, encoding="utf-8")
    return found_fig, missing_fig, dup_removed, converted_count, unresolved_about_here, unresolved_citation_tokens + len(unresolved)


def write_appendix_source_notes() -> None:
    APP_NOTES.write_text(
        "# Appendix Source Notes\n\n"
        "- Table 1 values are dataset means over the severe-imbalance slice, derived from:\n"
        "  - `reports/neural_mlp_objective_perturbation_summary.md`\n"
        "  - `reports/neural_mlp_objective_perturbation/legacy_benchmark_summary.csv`\n"
        "  - `reports/neural_mlp_objective_perturbation/legacy_threshold_sweep_summary.csv`\n"
        "  - `results/geometry_transition_analysis/geometry_transition_model_means.csv`\n"
        "  - `reports/neural_mlp/legacy_benchmark_summary.csv`\n"
        "  - `reports/neural_mlp/legacy_threshold_sweep_summary.csv`\n\n"
        "- Table A compact values are from persistence-remediated family summaries:\n"
        "  - `paper/manuscript_tables/table_allocator_family_summary_v2.md`\n"
        "  - `reports/neural_mlp/legacy_benchmark_summary.csv`\n"
        "  - `reports/neural_mlp/legacy_threshold_sweep_summary.csv`\n"
        "  - `reports/neural_mlp/prediction_space_occupancy_summary.csv`\n\n"
        "- Definitions:\n"
        "  - `Recovery = Recall@0.01 - Recall@0.50`.\n"
        "  - `Persistence` denotes threshold-survival persistence (`threshold_occupancy_persistence_mean`).\n",
        encoding="utf-8",
    )


def run_pandoc() -> tuple[bool, bool, list[str]]:
    warnings: list[str] = []
    html_ok = False
    pdf_ok = False
    pandoc = shutil.which("pandoc")
    if not pandoc:
        warnings.append("Pandoc not found on PATH; HTML/PDF not generated.")
        return html_ok, pdf_ok, warnings

    try:
        subprocess.run(
            [
                pandoc,
                RENDERED_MD.name,
                "--citeproc",
                "--bibliography",
                "../references.bib",
                "-s",
                "-o",
                RENDERED_HTML.name,
            ],
            check=True,
            cwd=str(OUT),
        )
        html_ok = True
    except subprocess.CalledProcessError as exc:
        warnings.append(f"HTML build failed: {exc}")

    typst = shutil.which("typst")
    if typst:
        try:
            subprocess.run(
                [
                    pandoc,
                    RENDERED_MD.name,
                    "--citeproc",
                    "--bibliography",
                    "../references.bib",
                    "-o",
                    RENDERED_PDF.name,
                    "--pdf-engine=typst",
                ],
                check=True,
                cwd=str(OUT),
            )
            pdf_ok = True
        except subprocess.CalledProcessError as exc:
            warnings.append(f"PDF build failed with typst: {exc}")
    else:
        warnings.append("Typst not found; skipped typst PDF build.")

    if not pdf_ok:
        try:
            subprocess.run(
                [
                    pandoc,
                    RENDERED_MD.name,
                    "--citeproc",
                    "--bibliography",
                    "../references.bib",
                    "-o",
                    RENDERED_PDF.name,
                ],
                check=True,
                cwd=str(OUT),
            )
            pdf_ok = True
        except subprocess.CalledProcessError:
            warnings.append("PDF build failed with default Pandoc PDF pipeline.")

    return html_ok, pdf_ok, warnings


def write_report(
    found_fig: list[str],
    missing_fig: list[str],
    dup_removed: int,
    converted_count: int,
    unresolved_count: int,
    unresolved_about_here: int,
    html_ok: bool,
    pdf_ok: bool,
    warnings: list[str],
) -> None:
    lines = [
        "# Submission Render Report",
        "",
        "## Generated Files",
        "",
        f"- `{RENDERED_MD}`",
        f"- `{RENDERED_HTML}` ({'success' if html_ok else 'failed/skipped'})",
        f"- `{RENDERED_PDF}` ({'success' if pdf_ok else 'failed/skipped'})",
        f"- `{TABLE_OUT / 'table_1_compact.md'}`",
        f"- `{TABLE_OUT / 'table_allocator_family_summary_compact.md'}`",
        f"- `{APP_NOTES}`",
        "",
        "## Citation Conversion Status",
        "",
        f"- Converted placeholders: {converted_count}",
        f"- Unresolved placeholders: {unresolved_count}",
        "",
        "## Equation Rendering Status",
        "",
        "- Reachability display equation converted to Pandoc display-math block (`$$ ... $$`).",
        "",
        "## Figure Status",
        "",
        f"- Embedded figures: {', '.join(found_fig) if found_fig else 'None'}",
        f"- Missing figures: {', '.join(missing_fig) if missing_fig else 'None'}",
        f"- Duplicate caption lines removed: {dup_removed}",
        f"- Unresolved `about here` placeholders: {unresolved_about_here}",
        "",
        "## Table Status",
        "",
        "- Compact tables generated and inserted into main manuscript draft.",
        "- Full source/definition notes moved to appendix source-notes file.",
        "",
        "## Build Status",
        "",
        f"- HTML: {'success' if html_ok else 'failed/skipped'}",
        f"- PDF: {'success' if pdf_ok else 'failed/skipped'}",
    ]
    if warnings:
        lines.extend(["", "## Errors/Warnings", ""])
        lines.extend([f"- {w}" for w in warnings])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t1, ta = compact_tables()
    write_appendix_source_notes()
    found_fig, missing_fig, dup_removed, conv_count, unresolved_about_here, unresolved_count = build_manuscript(t1, ta)
    html_ok, pdf_ok, warnings = run_pandoc()
    write_report(
        found_fig,
        missing_fig,
        dup_removed,
        conv_count,
        unresolved_count,
        unresolved_about_here,
        html_ok,
        pdf_ok,
        warnings,
    )


if __name__ == "__main__":
    main()
