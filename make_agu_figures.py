"""Combine the manuscript's multi-panel figures into one PDF per figure.

AGU requires that all parts of a figure arrive as a single file, so the panels
that are currently written as separate PDFs are laid out here on a common page
and labelled (a), (b), ... The panels keep their vector content: each source
page is placed on the output page with a scale/translate transform rather than
being rasterised.

    python make_agu_figures.py

Output goes to agu_figures/ as figure02.pdf, figure05.pdf, figure06.pdf,
figure08.pdf and figure09.pdf.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "agu_figures"

# Gap between panels and the margin reserved for the (a)/(b) labels, in points.
PANEL_GAP = 10.0
LABEL_HEIGHT = 13.0
LABEL_FONT_SIZE = 10.0


def _page_size(path: Path) -> tuple[float, float]:
    box = PdfReader(path).pages[0].mediabox
    return float(box.width), float(box.height)


def _background_stream(width: float, height: float) -> DecodedStreamObject:
    """An opaque white rectangle covering the page.

    Some panels are saved with a transparent background, which a viewer renders
    against whatever lies behind it. Painting the page white first makes the
    combined figure match how the individual panels appear in the manuscript.
    """
    stream = DecodedStreamObject()
    stream.set_data(
        f"q 1 1 1 rg 0 0 {width:.3f} {height:.3f} re f Q".encode("latin-1")
    )
    return stream


def _label_stream(labels: list[tuple[str, float, float]]) -> DecodedStreamObject:
    """Build a content stream drawing each label at its (x, y) baseline."""
    parts = ["q", "BT", f"/AGUHelv {LABEL_FONT_SIZE} Tf", "0 0 0 rg"]
    for text, x, y in labels:
        parts.append(f"1 0 0 1 {x:.3f} {y:.3f} Tm")
        parts.append(f"({text}) Tj")
    parts += ["ET", "Q"]

    stream = DecodedStreamObject()
    stream.set_data("\n".join(parts).encode("latin-1"))
    return stream


def _add_labels(page: PageObject, labels: list[tuple[str, float, float]]) -> None:
    """Overlay panel labels on the page using the base-14 Helvetica font."""
    if not labels:
        return

    font = DictionaryObject()
    font.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        }
    )
    fonts = DictionaryObject()
    fonts.update({NameObject("/AGUHelv"): font})
    resources = page[NameObject("/Resources")]
    if NameObject("/Font") in resources:
        resources[NameObject("/Font")][NameObject("/AGUHelv")] = font
    else:
        resources[NameObject("/Font")] = fonts

    writer = page.indirect_reference.pdf
    stream = writer._add_object(_label_stream(labels))
    contents = page.raw_get("/Contents")
    page[NameObject("/Contents")] = ArrayObject([contents, stream])


def combine(
    sources: list[Path],
    output: Path,
    *,
    columns: int,
    labels: list[str] | None = None,
    target_width: float | None = None,
) -> None:
    """Place the source PDFs on one page in a `columns`-wide grid.

    Every panel is scaled to the same column width so the row heights line up,
    which is what keeps the axis text at a consistent size across panels.
    """
    missing = [p for p in sources if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing panel(s): " + ", ".join(str(p) for p in missing)
        )

    sizes = [_page_size(p) for p in sources]
    rows = -(-len(sources) // columns)

    # Column width: the widest panel, unless the caller pins the page width.
    if target_width is None:
        col_width = max(w for w, _ in sizes)
    else:
        col_width = (target_width - PANEL_GAP * (columns - 1)) / columns

    scales = [col_width / w for w, _ in sizes]
    scaled = [(col_width, h * s) for (_, h), s in zip(sizes, scales)]

    label_space = LABEL_HEIGHT if labels else 0.0
    row_heights = []
    for r in range(rows):
        row = scaled[r * columns : (r + 1) * columns]
        row_heights.append(max(h for _, h in row) + label_space)

    page_width = col_width * columns + PANEL_GAP * (columns - 1)
    page_height = sum(row_heights) + PANEL_GAP * (rows - 1)

    page = PageObject.create_blank_page(width=page_width, height=page_height)
    page[NameObject("/Contents")] = _background_stream(page_width, page_height)
    drawn_labels: list[tuple[str, float, float]] = []

    for index, (source, scale, (sw, sh)) in enumerate(zip(sources, scales, scaled)):
        row, col = divmod(index, columns)

        # Rows are filled from the top of the page downwards.
        top = page_height - sum(row_heights[:row]) - PANEL_GAP * row
        x = col * (col_width + PANEL_GAP)

        # The label sits in its own strip at the top of the row; the panel is
        # centred in the space beneath it. Panels often carry an opaque white
        # background, so an overlapping label would be painted over.
        y = top - row_heights[row] + (row_heights[row] - label_space - sh) / 2.0

        panel = PdfReader(source).pages[0]
        box = panel.mediabox
        transform = (
            Transformation()
            .translate(-float(box.left), -float(box.bottom))
            .scale(scale, scale)
            .translate(x, y)
        )
        page.merge_transformed_page(panel, transform)

        if labels:
            drawn_labels.append((labels[index], x, top - LABEL_FONT_SIZE))

    # The page must belong to the writer before the label stream is attached,
    # so that the stream can be registered as an indirect object.
    writer = PdfWriter()
    page = writer.add_page(page)
    _add_labels(page, drawn_labels)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)

    print(f"Saved: {output.relative_to(ROOT)}  "
          f"({page_width:.0f} x {page_height:.0f} pt, {len(sources)} panels)")


FIGURES = [
    # Figure 2: weekly density (left) and GRACE altitude (right).
    dict(
        output="figure02.pdf",
        columns=2,
        labels=["(a)", "(b)"],
        sources=["rho_vs_msis_weekly_AGU.pdf", "figure_altvstime.pdf"],
    ),
    # Figure 5: on-track density on three illustrative days.
    dict(
        output="figure05.pdf",
        columns=1,
        labels=["(a)", "(b)", "(c)"],
        sources=[
            "figs/showcase_quiet_2009-05-03.pdf",
            "figs/showcase_storm_2015-03-17.pdf",
            "figs/showcase_disturbed_2016-02-16.pdf",
        ],
    ),
    # Figure 6: global off-track maps, baseline and model, with/without Swarm.
    dict(
        output="figure06.pdf",
        columns=2,
        labels=["(a)", "(b)", "(c)", "(d)"],
        sources=[
            "plot_grace_globalmsis.pdf",
            "plot_grace_pred.pdf",
            "plot_swarm_global_scale_line_msis.pdf",
            "plot_swarm_global_scale_line_pred.pdf",
        ],
    ),
    # Figure 9: the same three days at one- and three-day update latency.
    dict(
        output="figure09.pdf",
        columns=1,
        labels=["(a)", "(b)", "(c)"],
        sources=[
            "figs/showcase_lead3_2009-05-03.pdf",
            "figs/showcase_lead3_2015-03-17.pdf",
            "figs/showcase_lead3_2016-02-16.pdf",
        ],
    ),
]

# Figures that are already single files are copied across unchanged so that
# agu_figures/ holds the complete set for upload.
SINGLE_PANEL = [
    ("feature_importance_xgb_model_v8_storm_ap_2002train.pdf", "figure03.pdf"),
    ("parity_h1.pdf", "figure04.pdf"),
    # plot_tuning_summary.py draws both panels on one figure.
    ("tuning_v13_tec3h_depth3_10/tuning_summary.pdf", "figure07.pdf"),
    # plot_skill_regimes.py already stacks the three regimes into one figure.
    ("figs/skill_regimes_dr1_h1.pdf", "figure08.pdf"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for spec in FIGURES:
        combine(
            [ROOT / s for s in spec["sources"]],
            OUTPUT_DIR / spec["output"],
            columns=spec["columns"],
            labels=spec["labels"],
        )

    for source_name, output_name in SINGLE_PANEL:
        source = ROOT / source_name
        if not source.is_file():
            print(f"Skipped: {source_name} not found")
            continue
        (OUTPUT_DIR / output_name).write_bytes(source.read_bytes())
        print(f"Copied: agu_figures/{output_name}  (from {source_name})")


if __name__ == "__main__":
    main()
