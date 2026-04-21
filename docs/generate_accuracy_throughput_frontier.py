#!/usr/bin/env python3
"""Generate the standalone HIGGS accuracy-versus-throughput frontier figure."""

from __future__ import annotations

import argparse
import html
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AccuracyRow:
    model: str
    hidden_dim: int
    num_layers: int
    accuracy_pct: float
    roc_auc: float
    pr_auc: float
    log_loss: float


@dataclass(frozen=True)
class ThroughputRow:
    configuration: str
    hidden_dim: int
    num_layers: int
    wall_mps: float
    kernel_mps: float | None


def parse_args() -> argparse.Namespace:
    docs_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Create the retained HIGGS/XDNA2 accuracy-throughput frontier figure "
            "from the tables in docs/whitepaper.tex."
        )
    )
    parser.add_argument("--whitepaper", type=Path, default=docs_dir / "whitepaper.tex")
    parser.add_argument(
        "--output",
        type=Path,
        default=docs_dir / "accuracy_throughput_frontier.svg",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=docs_dir / "accuracy_throughput_frontier.pdf",
        help="Optional PDF output rendered from the generated SVG when a converter is available.",
    )
    return parser.parse_args()


def strip_table_row(line: str) -> str:
    line = line.strip()
    if line.endswith(r"\\"):
        return line[:-2].strip()
    return line


def parse_accuracy_rows(tex: str) -> list[AccuracyRow]:
    rows: list[AccuracyRow] = []
    for raw_line in tex.splitlines():
        line = raw_line.strip()
        if "\\%" not in line or "$H=" not in line or "&" not in line:
            continue
        clean = strip_table_row(line)
        parts = [part.strip() for part in clean.split("&")]
        if len(parts) != 5:
            continue
        # Accuracy rows have the percent sign in column 1 (e.g. "76.56\%");
        # throughput rows put the percent sign in the final column. Skip anything
        # whose second cell is not a percentage.
        if r"\%" not in parts[1]:
            continue
        model = parts[0]
        match = re.search(r"H=(\d+).+L=(\d+)", model)
        if not match:
            continue
        rows.append(
            AccuracyRow(
                model=model,
                hidden_dim=int(match.group(1)),
                num_layers=int(match.group(2)),
                accuracy_pct=float(parts[1].replace(r"\%", "")),
                roc_auc=float(parts[2]),
                pr_auc=float(parts[3]),
                log_loss=float(parts[4]),
            )
        )
    return rows


def parse_throughput_rows(tex: str) -> list[ThroughputRow]:
    """Parse rows from the async/sync throughput table.

    Expected row shape is either:
      configuration & sync-wall & async-wall & kernel & wall/kernel
    or:
      configuration & sync-wall & async-wall

    The frontier figure only uses the async wall-throughput column, but the
    parser also tolerates the wall-only table shape used during intermediate
    documentation edits.
    """
    rows: list[ThroughputRow] = []
    num_re = re.compile(r"([\d.]+)M")
    for raw_line in tex.splitlines():
        line = raw_line.strip()
        if "$H=" not in line or "&" not in line or "M" not in line:
            continue
        clean = strip_table_row(line)
        parts = [part.strip() for part in clean.split("&")]
        if len(parts) not in (3, 5):
            continue
        configuration = parts[0]
        match = re.search(r"H=(\d+).+L=(\d+)", configuration)
        if not match:
            continue
        async_wall = num_re.search(parts[2])
        if async_wall is None:
            continue
        kernel = num_re.search(parts[3]) if len(parts) == 5 else None
        rows.append(
            ThroughputRow(
                configuration=configuration,
                hidden_dim=int(match.group(1)),
                num_layers=int(match.group(2)),
                wall_mps=float(async_wall.group(1)),
                kernel_mps=float(kernel.group(1)) if kernel is not None else None,
            )
        )
    return rows


def find_accuracy(
    rows: list[AccuracyRow], hidden_dim: int, num_layers: int, model_fragment: str
) -> AccuracyRow:
    for row in rows:
        if (
            row.hidden_dim == hidden_dim
            and row.num_layers == num_layers
            and model_fragment in row.model
        ):
            return row
    raise ValueError(
        f"Could not find accuracy row for H={hidden_dim}, L={num_layers}, {model_fragment!r}"
    )


def find_throughput(
    rows: list[ThroughputRow], hidden_dim: int, num_layers: int
) -> ThroughputRow:
    for row in rows:
        if row.hidden_dim == hidden_dim and row.num_layers == num_layers:
            return row
    raise ValueError(
        f"Could not find throughput row for H={hidden_dim}, L={num_layers}"
    )


def escape(text: str) -> str:
    return html.escape(text, quote=True)


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    value = start
    while value <= stop + 1e-9:
        values.append(round(value, 10))
        value += step
    return values


def measured_point_label(hidden_dim: int, num_layers: int) -> str:
    return f"H={hidden_dim}, L={num_layers}"


def render_pdf(svg_path: Path, pdf_path: Path) -> bool:
    if tool := shutil.which("rsvg-convert"):
        subprocess.run(
            [tool, "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
            check=True,
        )
        return True
    if tool := shutil.which("inkscape"):
        subprocess.run(
            [
                tool,
                str(svg_path),
                "--export-type=pdf",
                f"--export-filename={pdf_path}",
            ],
            check=True,
        )
        return True
    if tool := shutil.which("convert"):
        subprocess.run([tool, str(svg_path), str(pdf_path)], check=True)
        return True
    return False


def render_svg(
    throughput_point: tuple[AccuracyRow, ThroughputRow],
    manual_point: tuple[AccuracyRow, ThroughputRow],
    best_accuracy_point: tuple[AccuracyRow, ThroughputRow],
) -> str:
    width = 1200
    height = 690
    chart_left = 92
    chart_top = 132
    chart_width = 710
    chart_height = 424
    chart_right = chart_left + chart_width
    chart_bottom = chart_top + chart_height
    panel_x = chart_right + 40
    panel_width = width - panel_x - 34

    throughput_entry = {
        "name": measured_point_label(
            throughput_point[0].hidden_dim, throughput_point[0].num_layers
        ),
        "accuracy": throughput_point[0].accuracy_pct,
        "roc_auc": throughput_point[0].roc_auc,
        "throughput": throughput_point[1].wall_mps,
        "color": "#1f77b4",
        "emphasis": "Best throughput",
    }
    manual_entry = {
        "name": measured_point_label(
            manual_point[0].hidden_dim, manual_point[0].num_layers
        ),
        "accuracy": manual_point[0].accuracy_pct,
        "roc_auc": manual_point[0].roc_auc,
        "throughput": manual_point[1].wall_mps,
        "color": "#2ca02c",
        "emphasis": "Best full-data manual",
    }
    best_accuracy_entry = {
        "name": measured_point_label(
            best_accuracy_point[0].hidden_dim, best_accuracy_point[0].num_layers
        ),
        "accuracy": best_accuracy_point[0].accuracy_pct,
        "roc_auc": best_accuracy_point[0].roc_auc,
        "throughput": best_accuracy_point[1].wall_mps,
        "color": "#d95f02",
        "emphasis": "Best accuracy",
    }

    measured = sorted(
        [throughput_entry, manual_entry, best_accuracy_entry],
        key=lambda point: point["throughput"],
    )

    x_values = [point["throughput"] for point in measured]
    y_values = [point["accuracy"] for point in measured]
    x_min = math.floor((min(x_values) - 0.15) * 10) / 10
    x_max = math.ceil((max(x_values) + 0.15) * 10) / 10
    y_min = math.floor((min(y_values) - 0.25) * 2) / 2
    y_max = math.ceil((max(y_values) + 0.2) * 2) / 2

    def x_to_svg(value: float) -> float:
        return chart_left + ((value - x_min) / (x_max - x_min)) * chart_width

    def y_to_svg(value: float) -> float:
        return chart_bottom - ((value - y_min) / (y_max - y_min)) * chart_height

    x_ticks = frange(x_min, x_max, 0.2)
    y_ticks = frange(y_min, y_max, 0.5)
    svg: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        'text { font-family: "DejaVu Sans", Arial, sans-serif; fill: #202124; }',
        ".title { font-size: 24px; font-weight: 700; }",
        ".subtitle { font-size: 14px; fill: #4d5156; }",
        ".axis-label { font-size: 15px; font-weight: 600; }",
        ".tick { font-size: 12px; fill: #5f6368; }",
        ".note { font-size: 13px; fill: #3c4043; }",
        ".small { font-size: 11px; fill: #5f6368; }",
        ".panel-title { font-size: 16px; font-weight: 700; }",
        ".label { font-size: 13px; font-weight: 600; }",
        "</style>",
        '<text class="title" x="40" y="42">HIGGS accuracy vs. wall throughput on XDNA2</text>',
        (
            '<text class="subtitle" x="40" y="66">'
            'Measured points use the B=8 async wall-throughput rows from Table 4 and the retained '
            'accuracy rows from Table 3.'
            "</text>"
        ),
        (
            '<text class="subtitle" x="40" y="86">'
            'All three retained operating points now have measured hardware throughput.'
            "</text>"
        ),
    ]

    for tick in x_ticks:
        x = x_to_svg(tick)
        svg.append(
            f'<line x1="{x:.2f}" y1="{chart_top}" x2="{x:.2f}" y2="{chart_bottom}" '
            'stroke="#eceff1" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="tick" x="{x:.2f}" y="{chart_bottom + 24}" text-anchor="middle">'
            f"{tick:.1f}</text>"
        )

    for tick in y_ticks:
        y = y_to_svg(tick)
        svg.append(
            f'<line x1="{chart_left}" y1="{y:.2f}" x2="{chart_right}" y2="{y:.2f}" '
            'stroke="#eceff1" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="tick" x="{chart_left - 12}" y="{y + 4:.2f}" text-anchor="end">'
            f"{tick:.1f}%</text>"
        )

    svg.extend(
        [
            (
                f'<line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_bottom}" '
                'stroke="#5f6368" stroke-width="1.5"/>'
            ),
            (
                f'<line x1="{chart_left}" y1="{chart_bottom}" x2="{chart_right}" y2="{chart_bottom}" '
                'stroke="#5f6368" stroke-width="1.5"/>'
            ),
        ]
    )

    line_points = " ".join(
        f"{x_to_svg(point['throughput']):.2f},{y_to_svg(point['accuracy']):.2f}"
        for point in measured
    )
    svg.append(
        f'<polyline fill="none" stroke="#5f6368" stroke-width="2.5" points="{line_points}" opacity="0.7"/>'
    )

    label_positions = {
        "H=64, L=32": (20, -34),
        "H=32, L=30": (-188, 18),
        "H=32, L=32": (20, -34),
    }
    label_box_sizes = {
        "H=64, L=32": (206, 60),
        "H=32, L=30": (220, 60),
        "H=32, L=32": (206, 60),
    }
    for point in measured:
        cx = x_to_svg(point["throughput"])
        cy = y_to_svg(point["accuracy"])
        svg.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7.5" fill="{point["color"]}" '
            'stroke="#ffffff" stroke-width="2"/>'
        )
        dx, dy = label_positions[point["name"]]
        label_x = cx + dx
        label_y = cy + dy
        box_width, box_height = label_box_sizes[point["name"]]
        svg.append(
            f'<rect x="{label_x - 10:.2f}" y="{label_y - 20:.2f}" width="{box_width}" '
            f'height="{box_height}" rx="8" fill="#ffffff" fill-opacity="0.9"/>'
        )
        svg.append(
            f'<text class="label" x="{label_x:.2f}" y="{label_y:.2f}">{escape(point["name"])}</text>'
        )
        svg.append(
            f'<text class="note" x="{label_x:.2f}" y="{label_y + 18:.2f}">'
            f'{point["throughput"]:.2f}M/s, {point["accuracy"]:.2f}% acc'
            "</text>"
        )
        svg.append(
            f'<text class="small" x="{label_x:.2f}" y="{label_y + 34:.2f}">'
            f'{escape(point["emphasis"])} · ROC AUC {point["roc_auc"]:.4f}'
            "</text>"
        )

    panel_height = 286
    svg.extend(
        [
            f'<text class="axis-label" x="{chart_left + chart_width / 2:.2f}" y="{height - 46}" text-anchor="middle">',
            "Wall throughput (million samples / s)",
            "</text>",
            (
                f'<text class="axis-label" x="28" y="{chart_top + chart_height / 2:.2f}" '
                f'transform="rotate(-90 28 {chart_top + chart_height / 2:.2f})" text-anchor="middle">'
                "Test accuracy"
                "</text>"
            ),
            (
                f'<rect x="{panel_x}" y="{chart_top}" width="{panel_width}" height="{panel_height}" '
                'rx="14" fill="#f8f9fa" stroke="#dadce0"/>'
            ),
            f'<text class="panel-title" x="{panel_x + 18}" y="{chart_top + 30}">Retained operating points</text>',
        ]
    )

    panel_entries = [
        (
            throughput_entry["color"],
            f'{throughput_entry["name"]} (throughput point)',
            [
                f'{throughput_entry["accuracy"]:.2f}% acc · {throughput_entry["throughput"]:.2f}M/s wall',
                f'ROC AUC {throughput_entry["roc_auc"]:.4f}',
            ],
        ),
        (
            manual_entry["color"],
            f'{manual_entry["name"]} (best full-data manual)',
            [
                f'{manual_entry["accuracy"]:.2f}% acc · {manual_entry["throughput"]:.2f}M/s wall',
                f'ROC AUC {manual_entry["roc_auc"]:.4f}',
            ],
        ),
        (
            best_accuracy_entry["color"],
            f'{best_accuracy_entry["name"]} tuned (best accuracy)',
            [
                f'{best_accuracy_entry["accuracy"]:.2f}% acc · {best_accuracy_entry["throughput"]:.2f}M/s wall',
                f'ROC AUC {best_accuracy_entry["roc_auc"]:.4f}',
            ],
        ),
    ]
    text_y = chart_top + 64
    for color, label, details in panel_entries:
        svg.append(
            f'<circle cx="{panel_x + 18}" cy="{text_y - 5}" r="5.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
        )
        text_x = panel_x + 34
        svg.append(
            f'<text class="note" x="{text_x}" y="{text_y}">{escape(label)}</text>'
        )
        text_y += 22
        for detail in details:
            svg.append(
                f'<text class="small" x="{text_x}" y="{text_y}">{escape(detail)}</text>'
            )
            text_y += 16
        text_y += 14

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def main() -> int:
    args = parse_args()
    tex = args.whitepaper.read_text(encoding="utf-8")
    accuracy_rows = parse_accuracy_rows(tex)
    throughput_rows = parse_throughput_rows(tex)

    throughput_point = (
        find_accuracy(accuracy_rows, 32, 30, "full-NPU"),
        find_throughput(throughput_rows, 32, 30),
    )
    manual_point = (
        find_accuracy(accuracy_rows, 32, 32, "full-data (20 epochs)"),
        find_throughput(throughput_rows, 32, 32),
    )
    best_accuracy = (
        find_accuracy(accuracy_rows, 64, 32, "Validation-selected tuning run"),
        find_throughput(throughput_rows, 64, 32),
    )

    svg = render_svg(throughput_point, manual_point, best_accuracy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.pdf_output:
        if render_pdf(args.output, args.pdf_output):
            print(f"Wrote {args.pdf_output}")
        else:
            print(
                "Warning: could not render PDF copy because neither rsvg-convert nor inkscape is available."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
