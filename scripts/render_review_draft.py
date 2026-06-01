from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT_DIR = PAPER / "rendered_review_draft"


FIGURES = {
    "Figure 0": "figure_0_operational_accessibility_lens.png",
    "Figure 1": "figure_1_reachability_mean_transition.png",
    "Figure 2": "figure_2_reachability_by_dataset.png",
    "Figure 3": "figure_3_elasticity_interval_heatmap.png",
    "Figure 4": "figure_4_support_vs_persistence.png",
    "Figure 5": "figure_5_calibration_geometry_deltas.png",
    "Figure 7": "figure_accessibility_regime_map_variant_a_v2.png",
}

TABLES = {
    "Table 1": PAPER / "manuscript_tables" / "table_1_ranking_vs_accessibility.md",
    "Table A": PAPER / "manuscript_tables" / "table_allocator_family_summary_v2.md",
}


def build_review_markdown(src_md: Path, out_md: Path) -> tuple[list[str], list[str], int]:
    text = src_md.read_text(encoding="utf-8")
    missing_assets: list[str] = []
    warnings: list[str] = []

    # Replace figure placeholders with inline image blocks.
    for fig_label, filename in FIGURES.items():
        placeholder = f"[{fig_label} about here]"
        fig_path = PAPER / "manuscript_figures" / filename
        if not fig_path.exists():
            missing_assets.append(str(fig_path))
            continue

        rel = Path("..") / "manuscript_figures" / filename
        block = f"![{fig_label}]({rel.as_posix()}){{ width=100% }}"
        text = text.replace(placeholder, block)

    # Replace table placeholders with table markdown contents.
    for table_label, table_path in TABLES.items():
        placeholder = f"[{table_label} about here]"
        if not table_path.exists():
            missing_assets.append(str(table_path))
            continue

        table_md = table_path.read_text(encoding="utf-8").strip()
        # Trim title header if present to avoid duplicate heading in manuscript flow.
        table_md = re.sub(r"^#\s+.*\n+", "", table_md, count=1)
        text = text.replace(placeholder, table_md)

    unresolved_citations = len(re.findall(r"\[CITATION:\s*[^\]]+\]", text))

    out_md.write_text(text, encoding="utf-8")
    return missing_assets, warnings, unresolved_citations


def run_pandoc(input_md: Path, output_html: Path, output_pdf: Path) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    generated: list[str] = []

    pandoc = shutil.which("pandoc")
    if not pandoc:
        # Fallback: try Python markdown-based HTML generation.
        try:
            import markdown  # type: ignore

            md_text = input_md.read_text(encoding="utf-8")
            body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
            html = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>manuscript_v0_9a_review</title>"
                "<style>body{max-width:960px;margin:2rem auto;padding:0 1rem;font-family:Georgia,serif;line-height:1.55;}"
                "img{max-width:100%;height:auto;} table{border-collapse:collapse;width:100%;}"
                "th,td{border:1px solid #999;padding:6px;vertical-align:top;}"
                "h1,h2,h3{line-height:1.25;} .toc{border:1px solid #ddd;padding:0.75rem 1rem;background:#fafafa;}</style>"
                "</head><body><h1>Review Draft</h1>"
                "<p><em>Auto-rendered fallback HTML (Pandoc unavailable).</em></p>"
                f"{body}</body></html>"
            )
            output_html.write_text(html, encoding="utf-8")
            generated.append(str(output_html))
            warnings.append("Pandoc not found; used Python markdown fallback for HTML. TOC/equations may be degraded.")
        except Exception:
            md_text = input_md.read_text(encoding="utf-8")
            escaped = (
                md_text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            html = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>manuscript_v0_9a_review</title>"
                "<style>body{max-width:960px;margin:2rem auto;padding:0 1rem;font-family:Georgia,serif;}"
                "pre{white-space:pre-wrap;line-height:1.5;background:#fafafa;border:1px solid #ddd;padding:1rem;}"
                "</style></head><body><h1>Review Draft</h1>"
                "<p><em>Fallback HTML (raw markdown view; Pandoc and markdown package unavailable).</em></p>"
                f"<pre>{escaped}</pre></body></html>"
            )
            output_html.write_text(html, encoding="utf-8")
            generated.append(str(output_html))
            warnings.append("Pandoc not found; generated fallback raw-markdown HTML.")
        warnings.append("PDF rendering failed (Pandoc unavailable).")
        return generated, warnings

    # HTML with TOC and math rendering.
    html_cmd = [
        pandoc,
        str(input_md),
        "-s",
        "--toc",
        "--toc-depth=3",
        "--mathjax",
        "-o",
        str(output_html),
    ]
    try:
        subprocess.run(html_cmd, check=True, cwd=str(ROOT))
        generated.append(str(output_html))
    except subprocess.CalledProcessError as exc:
        warnings.append(f"HTML rendering failed: {exc}")

    # PDF render: try default engine first, then wkhtmltopdf fallback.
    pdf_ok = False
    pdf_cmd_default = [pandoc, str(input_md), "-s", "-o", str(output_pdf)]
    try:
        subprocess.run(pdf_cmd_default, check=True, cwd=str(ROOT))
        generated.append(str(output_pdf))
        pdf_ok = True
    except subprocess.CalledProcessError:
        pass

    if not pdf_ok:
        wkhtml = shutil.which("wkhtmltopdf")
        if wkhtml:
            pdf_cmd_wk = [
                pandoc,
                str(input_md),
                "-s",
                "--pdf-engine=wkhtmltopdf",
                "-o",
                str(output_pdf),
            ]
            try:
                subprocess.run(pdf_cmd_wk, check=True, cwd=str(ROOT))
                generated.append(str(output_pdf))
                pdf_ok = True
            except subprocess.CalledProcessError as exc:
                warnings.append(f"PDF rendering failed with wkhtmltopdf: {exc}")

    if not pdf_ok:
        warnings.append("PDF rendering failed (no usable PDF engine found by Pandoc).")

    # Optional docx.
    docx_path = output_html.with_suffix(".docx")
    try:
        subprocess.run([pandoc, str(input_md), "-s", "-o", str(docx_path)], check=True, cwd=str(ROOT))
        generated.append(str(docx_path))
    except subprocess.CalledProcessError:
        warnings.append("Optional DOCX rendering failed.")

    return generated, warnings


def write_report(
    report_path: Path,
    generated: list[str],
    missing_assets: list[str],
    unresolved_citations: int,
    warnings: list[str],
) -> None:
    lines = [
        "# Render Report",
        "",
        "## Generated Files",
        "",
    ]
    if generated:
        lines.extend(f"- `{g}`" for g in generated)
    else:
        lines.append("- None")

    lines.extend(["", "## Missing Figures/Tables", ""])
    if missing_assets:
        lines.extend(f"- `{m}`" for m in missing_assets)
    else:
        lines.append("- None")

    lines.extend(["", "## Unresolved Citation Placeholders", ""])
    lines.append(f"- Count: {unresolved_citations}")

    lines.extend(["", "## Rendering Warnings", ""])
    if warnings:
        lines.extend(f"- {w}" for w in warnings)
    else:
        lines.append("- None")

    lines.extend(["", "## Recommended Manual Fixes", ""])
    fixes = []
    if unresolved_citations > 0:
        fixes.append("Resolve or replace remaining `[CITATION: ...]` placeholders before submission copy-edit.")
    if missing_assets:
        fixes.append("Add missing figure/table source files and rerun render script.")
    if any("PDF rendering failed" in w for w in warnings):
        fixes.append("Install a Pandoc-compatible PDF engine (TeX or wkhtmltopdf) and rerun.")
    if not fixes:
        fixes.append("No blocking fixes detected; perform visual QA on HTML/PDF pagination and captions.")
    lines.extend(f"- {f}" for f in fixes)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src = PAPER / "manuscript_v0_9a.md"
    out_md = OUT_DIR / "manuscript_v0_9a_review.md"
    out_html = OUT_DIR / "manuscript_v0_9a_review.html"
    out_pdf = OUT_DIR / "manuscript_v0_9a_review.pdf"
    report = OUT_DIR / "render_report.md"

    missing_assets, md_warnings, unresolved = build_review_markdown(src, out_md)
    generated = [str(out_md)]

    rendered, render_warnings = run_pandoc(out_md, out_html, out_pdf)
    generated.extend(rendered)

    all_warnings = md_warnings + render_warnings
    write_report(report, generated, missing_assets, unresolved, all_warnings)


if __name__ == "__main__":
    main()
