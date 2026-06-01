from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "rendered_review_draft"

SRC = PAPER / "manuscript_v0_9a.md"
OUT_MD = OUT / "manuscript_v0_9a_rendered.md"
OUT_HTML = OUT / "manuscript_v0_9a_rendered.html"
OUT_PDF = OUT / "manuscript_v0_9a_rendered.pdf"
REPORT = OUT / "render_report.md"


FIGURE_MAP = {
    "Figure 0": "figure_0_operational_accessibility_lens.png",
    "Figure 1": "figure_1_reachability_mean_transition.png",
    "Figure 2": "figure_2_reachability_by_dataset.png",
    "Figure 3": "figure_3_elasticity_interval_heatmap.png",
    "Figure 4": "figure_4_support_vs_persistence.png",
    "Figure 5": "figure_5_calibration_geometry_deltas.png",
    "Figure 7": "figure_accessibility_regime_map_variant_a_v2.png",
}

TABLE_MAP = {
    "Table 1": PAPER / "manuscript_tables" / "table_1_ranking_vs_accessibility.md",
    "Table A": PAPER / "manuscript_tables" / "table_allocator_family_summary_v2.md",
}


def extract_figure_captions(text: str) -> dict[str, str]:
    caps: dict[str, str] = {}
    for m in re.finditer(r"\*\*(Figure\s+\d+)\:\*\*\s*(.+)", text):
        label = m.group(1).strip()
        caption = m.group(2).strip()
        caps[label] = caption
    return caps


def strip_first_heading(md: str) -> str:
    return re.sub(r"^#\s+.*\n+", "", md.strip(), count=1)


def expand_markdown() -> tuple[list[str], list[str], list[str], list[str], int]:
    text = SRC.read_text(encoding="utf-8")
    captions = extract_figure_captions(text)

    figures_found: list[str] = []
    figures_missing: list[str] = []
    tables_found: list[str] = []
    tables_missing: list[str] = []

    for label, filename in FIGURE_MAP.items():
        placeholder = f"[{label} about here]"
        fig_path = PAPER / "manuscript_figures" / filename
        if fig_path.exists():
            caption = captions.get(label, f"{label}.")
            img_rel = (Path("..") / "manuscript_figures" / filename).as_posix()
            img_block = f"![{label}. {caption}]({img_rel})\n"
            text = text.replace(placeholder, img_block)
            figures_found.append(filename)
        else:
            figures_missing.append(filename)

    for label, table_path in TABLE_MAP.items():
        placeholder = f"[{label} about here]"
        if table_path.exists():
            tmd = strip_first_heading(table_path.read_text(encoding="utf-8"))
            text = text.replace(placeholder, tmd)
            tables_found.append(str(table_path.name))
        else:
            tables_missing.append(str(table_path))

    unresolved_about_here = len(re.findall(r"\[(Figure\s+\d+|Table\s+[A0-9]+)\s+about\s+here\]", text))
    OUT_MD.write_text(text, encoding="utf-8")
    return figures_found, figures_missing, tables_found, tables_missing, unresolved_about_here


def run_builds() -> tuple[bool, bool, list[str]]:
    warnings: list[str] = []
    html_ok = False
    pdf_ok = False
    pandoc = shutil.which("pandoc")

    if pandoc:
        try:
            subprocess.run(
                [pandoc, OUT_MD.name, "-s", "-o", OUT_HTML.name],
                check=True,
                cwd=str(OUT),
            )
            html_ok = True
        except subprocess.CalledProcessError as exc:
            warnings.append(f"HTML build failed via Pandoc: {exc}")

        # Prefer typst if present
        typst = shutil.which("typst")
        if typst:
            try:
                subprocess.run(
                    [pandoc, OUT_MD.name, "-o", OUT_PDF.name, "--pdf-engine=typst"],
                    check=True,
                    cwd=str(OUT),
                )
                pdf_ok = True
            except subprocess.CalledProcessError as exc:
                warnings.append(f"PDF build failed with typst engine: {exc}")
        else:
            warnings.append("Typst not found; skipping typst PDF build.")

        if not pdf_ok:
            try:
                subprocess.run([pandoc, OUT_MD.name, "-o", OUT_PDF.name], check=True, cwd=str(OUT))
                pdf_ok = True
            except subprocess.CalledProcessError:
                warnings.append("PDF build failed with default Pandoc PDF pipeline.")
    else:
        warnings.append("Pandoc not found on PATH; HTML/PDF builds skipped.")

    return html_ok, pdf_ok, warnings


def write_report(
    figures_found: list[str],
    figures_missing: list[str],
    tables_found: list[str],
    tables_missing: list[str],
    unresolved_about_here: int,
    html_ok: bool,
    pdf_ok: bool,
    warnings: list[str],
) -> None:
    lines = [
        "# Render Report",
        "",
        "## Generated Artifacts",
        "",
        f"- `{OUT_MD}`",
        f"- `{OUT_HTML}` ({'success' if html_ok else 'not generated'})",
        f"- `{OUT_PDF}` ({'success' if pdf_ok else 'not generated'})",
        "",
        "## Figures Found",
        "",
    ]
    lines.extend([f"- `{f}`" for f in figures_found] or ["- None"])
    lines.extend(["", "## Figures Missing", ""])
    lines.extend([f"- `{f}`" for f in figures_missing] or ["- None"])

    lines.extend(["", "## Tables Found", ""])
    lines.extend([f"- `{t}`" for t in tables_found] or ["- None"])
    lines.extend(["", "## Tables Missing", ""])
    lines.extend([f"- `{t}`" for t in tables_missing] or ["- None"])

    lines.extend(["", "## Placeholder Expansion Check", ""])
    lines.append(f"- Unresolved `[... about here]` placeholders: {unresolved_about_here}")

    lines.extend(["", "## Build Status", ""])
    lines.append(f"- HTML build: {'success' if html_ok else 'failed/skipped'}")
    lines.append(f"- PDF build: {'success' if pdf_ok else 'failed/skipped'}")

    lines.extend(["", "## Rendering Warnings", ""])
    lines.extend([f"- {w}" for w in warnings] or ["- None"])

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (
        figures_found,
        figures_missing,
        tables_found,
        tables_missing,
        unresolved_about_here,
    ) = expand_markdown()
    html_ok, pdf_ok, warnings = run_builds()
    write_report(
        figures_found,
        figures_missing,
        tables_found,
        tables_missing,
        unresolved_about_here,
        html_ok,
        pdf_ok,
        warnings,
    )


if __name__ == "__main__":
    main()
