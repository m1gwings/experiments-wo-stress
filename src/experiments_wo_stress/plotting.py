"""Line figures and editable PGFPlots sources generated from saved results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import Summary, _canonical, _load_class, _safe_name, analyze

_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
_LABELS = {"standard_error": "standard error", "std": "sample standard deviation"}


def _text(value: Any) -> str:
    return value if isinstance(value, str) else _canonical(value)


def _line_label(summary: Summary, color: str | None, panel: str | None) -> str:
    if color:
        return _text(summary.labels[color])
    return (
        ", ".join(f"{key}={_text(value)}" for key, value in summary.labels.items() if key != panel)
        or summary.metric
    )


def _panels(summaries: list[Summary], figure: dict[str, Any]) -> list[tuple[str, list[Summary]]]:
    panel = figure.get("panel")
    color = figure.get("color")
    grouped: dict[str, tuple[str, list[Summary]]] = {}
    seen: set[tuple[str, str]] = set()
    for summary in summaries:
        for label in (panel, color):
            if label is not None and label not in summary.labels:
                raise ValueError(f"Figure label {label!r} is not in aggregator.group_by")
        panel_key = _canonical(summary.labels[panel]) if panel else ""
        panel_label = (
            figure.get("panel_label", panel.rsplit(".", 1)[-1].replace("_", " ")) if panel else ""
        )
        title = f"{panel_label} = {_text(summary.labels[panel])}" if panel else ""
        label = _line_label(summary, color, panel)
        if (panel_key, label) in seen:
            raise ValueError(
                "Figure has multiple curves with the same panel and color; "
                "include differing group labels in the figure or use a custom plotter"
            )
        seen.add((panel_key, label))
        grouped.setdefault(panel_key, (title, []))[1].append(summary)
    return [grouped[key] for key in sorted(grouped)]


def _validate_line_figure(figure: dict[str, Any], summaries: list[Summary]) -> None:
    allowed = {
        "type",
        "metric",
        "x",
        "color",
        "panel",
        "panel_label",
        "formats",
        "name",
        "xlabel",
        "ylabel",
        "title",
        "xscale",
        "yscale",
    }
    unknown = set(figure) - allowed
    if unknown:
        raise ValueError(f"Unknown line figure options: {', '.join(sorted(unknown))}")
    for key in ("color", "panel", "panel_label", "x", "xlabel", "ylabel", "title"):
        if key in figure and not isinstance(figure[key], str):
            raise ValueError(f"Figure {key} must be a string")
    formats = figure.get("formats", ["pdf"])
    if (
        not isinstance(formats, list)
        or not formats
        or any(not isinstance(fmt, str) or fmt not in {"pdf", "jpg", "tikz"} for fmt in formats)
        or len(set(formats)) != len(formats)
    ):
        raise ValueError("Figure formats must be distinct entries from pdf, jpg, tikz")
    for scale in ("xscale", "yscale"):
        if figure.get(scale, "linear") not in {"linear", "log"}:
            raise ValueError(f"Figure {scale} must be linear or log")
    for summary in summaries:
        if figure.get("xscale") == "log" and np.any(summary.x <= 0):
            raise ValueError("A logarithmic x axis requires positive metric coordinates")
        if figure.get("yscale") == "log" and np.any(summary.mean <= 0):
            raise ValueError("A logarithmic y axis requires positive metric means")
    _panels(summaries, figure)


def plot(config: Any, output_dir: str | Path) -> list[Path]:
    """Recompute metrics from completed artifacts and export configured figures.

    Matplotlib is imported only for PDF or JPG exports. A custom plotter class
    receives ``params`` at construction and implements
    ``plot(summaries, figure, output_dir) -> iterable[Path]``.
    """
    figures = config.analysis.get("figures", [])
    if not isinstance(figures, list):
        raise ValueError("analysis.figures must be a list")
    summaries = analyze(config, output_dir)
    output = Path(output_dir) / "analysis" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    prepared = []
    names: set[str] = set()
    for index, figure in enumerate(figures):
        if not isinstance(figure, dict):
            raise ValueError("Each figure must be a mapping")
        metric = figure.get("metric")
        selected = [summary for summary in summaries if summary.metric == metric]
        if not selected:
            raise ValueError(f"Figure refers to unknown metric {metric!r}")
        name = _safe_name(figure.get("name", f"{metric}-{index + 1}"), "Figure name")
        if name in names:
            raise ValueError(f"Duplicate figure name {name!r}")
        names.add(name)
        if figure.get("type", "line") == "line":
            _validate_line_figure(figure, selected)
        else:
            unknown = set(figure) - {"type", "name", "metric", "params"}
            if unknown:
                raise ValueError(
                    "Custom figure options belong under params; unknown options: "
                    + ", ".join(sorted(unknown))
                )
        prepared.append((figure, selected, name))

    paths: list[Path] = []
    for figure, selected, name in prepared:
        if figure.get("type", "line") != "line":
            params = figure.get("params", {})
            if not isinstance(params, dict):
                raise ValueError("Custom plotter params must be a mapping")
            plotter = _load_class(figure["type"])(**params)
            if not callable(getattr(plotter, "plot", None)):
                raise TypeError(
                    "A custom plotter must implement plot(summaries, figure, output_dir)"
                )
            paths.extend(Path(path) for path in plotter.plot(selected, figure, output))
            continue
        formats = figure.get("formats", ["pdf"])
        if any(fmt in {"pdf", "jpg"} for fmt in formats):
            paths.extend(_matplotlib(selected, figure, output, name))
        if "tikz" in formats:
            target = output / f"{name}.tikz"
            target.write_text(_tikz(selected, figure), encoding="utf-8")
            paths.append(target)
    return paths


def _palette(summaries: list[Summary], figure: dict[str, Any]) -> dict[str, str]:
    labels = sorted(
        {_line_label(summary, figure.get("color"), figure.get("panel")) for summary in summaries}
    )
    return {label: _COLORS[index % len(_COLORS)] for index, label in enumerate(labels)}


def _band(summary: Summary, logarithmic: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = summary.mean - summary.uncertainty, summary.mean + summary.uncertainty
    valid = np.isfinite(lower) & np.isfinite(upper)
    if logarithmic:
        valid &= lower > 0
    return lower, upper, valid


def _matplotlib(
    summaries: list[Summary], figure: dict[str, Any], output: Path, name: str
) -> list[Path]:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError(
            "PDF/JPG export needs Matplotlib; install experiments-wo-stress[plot] "
            "or request only tikz"
        ) from exc
    panels = _panels(summaries, figure)
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    canvas_figure = Figure(figsize=(4.4 * columns, 3.5 * rows), constrained_layout=True)
    FigureCanvasAgg(canvas_figure)
    axes = canvas_figure.subplots(rows, columns, squeeze=False).ravel()
    palette = _palette(summaries, figure)
    for axis, (title, curves) in zip(axes, panels):
        for summary in curves:
            label = _line_label(summary, figure.get("color"), figure.get("panel"))
            color = palette[label]
            scalar = summary.x.size == 1
            axis.plot(
                summary.x,
                summary.mean,
                label=label,
                color=color,
                linewidth=1.6,
                marker="o" if scalar else None,
            )
            lower, upper, valid = _band(summary, figure.get("yscale") == "log")
            if summary.uncertainty_kind != "none" and valid.any():
                if scalar:
                    axis.errorbar(
                        summary.x,
                        summary.mean,
                        yerr=summary.uncertainty,
                        color=color,
                        capsize=3,
                        fmt="none",
                    )
                else:
                    axis.fill_between(summary.x, lower, upper, where=valid, color=color, alpha=0.18)
        axis.set(
            title=title,
            xlabel=figure.get("xlabel", figure.get("x", "step")),
            ylabel=figure.get("ylabel", figure["metric"]),
            xscale=figure.get("xscale", "linear"),
            yscale=figure.get("yscale", "linear"),
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize="small")
    for axis in axes[len(panels) :]:
        axis.set_visible(False)
    uncertainty_kind = summaries[0].uncertainty_kind
    title = figure.get("title", "")
    if uncertainty_kind != "none":
        note = f"Uncertainty: mean ± {_LABELS[uncertainty_kind]}; omitted for one repetition"
        title = f"{title}\n{note}" if title else note
    if title:
        canvas_figure.suptitle(title, fontsize=10)
    paths = []
    for fmt in figure.get("formats", ["pdf"]):
        if fmt in {"pdf", "jpg"}:
            target = output / f"{name}.{fmt}"
            canvas_figure.savefig(target, format=fmt, dpi=180)
            paths.append(target)
    canvas_figure.clear()
    return paths


def _latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\n": " ",
        "\r": " ",
    }
    return "".join(replacements.get(char, char) for char in value)


def _coordinates(x: np.ndarray, values: np.ndarray) -> str:
    return " ".join(f"({float(a):.17g},{float(b):.17g})" for a, b in zip(x, values))


def _tikz(summaries: list[Summary], figure: dict[str, Any]) -> str:
    panels = _panels(summaries, figure)
    columns = min(3, len(panels))
    palette = _palette(summaries, figure)
    colors = {label: f"ewscolor{index}" for index, label in enumerate(palette)}
    lines = [
        "% Generated from saved numerical results; compile with pdflatex or lualatex.",
        r"\documentclass[tikz,border=5pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\usepgfplotslibrary{groupplots,fillbetween}",
        r"\pgfplotsset{compat=1.18}",
    ]
    for label, hex_color in palette.items():
        lines.append(rf"\definecolor{{{colors[label]}}}{{HTML}}{{{hex_color[1:]}}}")
    lines.extend([r"\begin{document}", r"\begin{tikzpicture}"])
    lines.append(
        rf"\begin{{groupplot}}[group style={{group size={columns} by {math.ceil(len(panels) / columns)},"
        r"horizontal sep=1.8cm,vertical sep=2cm},width=7cm,height=5.5cm,"
        r"grid=major,grid style={gray!20},legend style={draw=none,font=\small}]"
    )
    uncertainty_kind = summaries[0].uncertainty_kind
    lines.append(f"% Uncertainty: {uncertainty_kind}; sample unit: independent run.")
    if uncertainty_kind != "none":
        lines.append("% Bands with nonpositive lower bounds are omitted on logarithmic y axes.")
    for panel_index, (title, curves) in enumerate(panels):
        if figure.get("title"):
            title = f"{figure['title']}: {title}" if title else figure["title"]
        options = [
            f"title={{{_latex(title)}}}",
            f"xlabel={{{_latex(figure.get('xlabel', figure.get('x', 'step')))}}}",
            f"ylabel={{{_latex(figure.get('ylabel', figure['metric']))}}}",
            f"xmode={figure.get('xscale', 'linear')}",
            f"ymode={figure.get('yscale', 'linear')}",
        ]
        if uncertainty_kind != "none":
            options.append(
                "extra description/.code={\\node[anchor=north,font=\\tiny] "
                "at (rel axis cs:0.5,-0.28) {Uncertainty: mean $\\pm$ "
                + _latex(_LABELS[uncertainty_kind])
                + "; omitted for one repetition};}"
            )
        lines.append(r"\nextgroupplot[" + ",".join(options) + "]")
        for curve_index, summary in enumerate(curves):
            label = _line_label(summary, figure.get("color"), figure.get("panel"))
            color = colors[label]
            lower, upper, valid = _band(summary, figure.get("yscale") == "log")
            if summary.uncertainty_kind != "none" and valid.any():
                # Split at invalid intervals so a band never bridges missing values.
                valid_indices = np.flatnonzero(valid)
                segments = np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) > 1) + 1)
                for segment_index, indices in enumerate(segments):
                    if indices.size < 2:
                        continue
                    prefix = f"p{panel_index}c{curve_index}s{segment_index}"
                    for suffix, values in (("lower", lower), ("upper", upper)):
                        lines.append(
                            rf"\addplot[draw=none,name path={prefix}{suffix},forget plot] coordinates {{"
                            + _coordinates(summary.x[indices], values[indices])
                            + "};"
                        )
                    lines.append(
                        rf"\addplot[{color},fill opacity=0.18,draw=none,forget plot] "
                        rf"fill between[of={prefix}lower and {prefix}upper];"
                    )
            style = f"{color},thick"
            coordinates = _coordinates(summary.x, summary.mean)
            if summary.x.size == 1:
                style += ",only marks,mark=*"
                if summary.uncertainty_kind != "none" and valid.any():
                    style += ",error bars/.cd,y dir=both,y explicit"
                    coordinates += f" +- (0,{float(summary.uncertainty[0]):.17g})"
            lines.append(rf"\addplot[{style}] coordinates {{" + coordinates + "};")
            lines.append(r"\addlegendentry{" + _latex(label) + "}")
    lines.extend([r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}", ""])
    return "\n".join(lines)
