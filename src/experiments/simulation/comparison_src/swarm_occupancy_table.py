#!/usr/bin/env python3
"""Emits the arena occupancy table as LaTeX."""

from __future__ import annotations

import argparse
from pathlib import Path

from occupancy_config import (
    A_BOT_CM2,
    ARENA_AREA_CM2,
    ARENA_SIDE_M,
    D_SAFE_TABLE_CM,
    HIGHLIGHT_SWARM_SIZES,
    ROBOT_A_CM,
    ROBOT_B_CM,
    buffer_area_cm2,
    default_swarm_sizes,
    n_max_at_d_safe,
    occupancy_pct,
)


def _fmt_area(x: float) -> str:
    if x >= 1000:
        return f"{x:,.1f}".replace(",", "{,}")
    return f"{x:.1f}"


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}"


def build_latex(
    d_safe_list_cm: list[float],
    swarm_sizes: list[int],
    highlight: set[int] | None = None,
) -> str:
    highlight = highlight or set(HIGHLIGHT_SWARM_SIZES)
    n_cols = 2 + 2 * len(d_safe_list_cm)
    col_spec = "c" + "cc" * (1 + len(d_safe_list_cm))

    buf_labels = []
    for d in d_safe_list_cm:
        nmax = n_max_at_d_safe(d)
        abuf = buffer_area_cm2(d)
        buf_labels.append(
            f"\\textbf{{Safety Buffer ($D_{{\\rm safe}} = {d:.0f}\\,\\mathrm{{cm}}$)}}"
        )

    header_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Swarm size, physical robot footprint, and safety buffer occupancies "
        rf"for different swarm sizes in a ${ARENA_SIDE_M:.1f}\,\mathrm{{m}} \times "
        rf"{ARENA_SIDE_M:.1f}\,\mathrm{{m}}$ (${ARENA_AREA_CM2:.0f}\,\mathrm{{cm}}^2$) arena. "
        rf"Physical footprint uses $A_{{\rm bot}} \approx {A_BOT_CM2:.2f}\,\mathrm{{cm}}^2$ "
        rf"(ellipse $a={ROBOT_A_CM:.0f}\,\mathrm{{cm}}$, $b={ROBOT_B_CM:.1f}\,\mathrm{{cm}}$). "
        "Exclusion zone areas use $A_{\\rm buffer} = \\pi (D_{\\rm safe}/2)^2$.}",
        r"\label{tab:swarm_density_and_occupancy}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    if len(d_safe_list_cm) == 1:
        d = d_safe_list_cm[0]
        nmax = n_max_at_d_safe(d)
        abuf = buffer_area_cm2(d)
        cap_extra = (
            rf"The exclusion zone areas are computed for $D_{{\rm safe}} = {d:.0f}\,\mathrm{{cm}}$ "
            rf"($A_{{\rm buffer}} \approx {abuf:.1f}\,\mathrm{{cm}}^2$, $N_{{\max}} = {nmax}$)."
        )
    else:
        parts = []
        for d in d_safe_list_cm:
            nmax = n_max_at_d_safe(d)
            abuf = buffer_area_cm2(d)
            parts.append(
                rf"$D_{{\rm safe}} = {d:.0f}\,\mathrm{{cm}}$ "
                rf"($A_{{\rm buffer}} \approx {abuf:.1f}\,\mathrm{{cm}}^2$, $N_{{\max}} = {nmax}$)"
            )
        cap_extra = "The exclusion zone areas are computed for " + " and ".join(parts) + "."

    header_lines[2] = header_lines[2].rstrip("}") + " " + cap_extra + "}"
    header_lines.append(
        r"& \multicolumn{2}{c}{\textbf{Physical Robot Footprint}} "
        + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{lbl}}}" for lbl in buf_labels
        )
    )
    header_lines.append(
        r"\cmidrule(lr){2-3}"
        + "".join(
            rf" \cmidrule(lr){{{4+2*i}-{5+2*i}}}" for i in range(len(d_safe_list_cm))
        )
    )
    hdr = (
        r"\textbf{Swarm Size} & \textbf{Total Area} & \textbf{Occupancy} "
    )
    for _ in d_safe_list_cm:
        hdr += r"& \textbf{Total Area} & \textbf{Occupancy} "
    header_lines.append(hdr.rstrip())
    header_lines.append(
        r"$N_b$ & ($\mathrm{cm}^2$) & (\%) "
        + r"& ($\mathrm{cm}^2$) & (\%) " * len(d_safe_list_cm)
    )
    header_lines.append(r"\midrule")

    body: list[str] = []
    for n in swarm_sizes:
        row_parts = [f"{n}"]
        a_bot = n * A_BOT_CM2
        row_parts += [_fmt_area(a_bot), _fmt_pct(occupancy_pct(a_bot))]
        for d in d_safe_list_cm:
            abuf = n * buffer_area_cm2(d)
            row_parts += [_fmt_area(abuf), _fmt_pct(occupancy_pct(abuf))]
        if n in highlight:
            row_parts = [rf"\textbf{{{c}}}" for c in row_parts]
            body.append(r"\rowcolor{blue!10} " + " & ".join(row_parts) + r" \\")
        else:
            body.append(" & ".join(row_parts) + r" \\")

    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(header_lines + body + footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate swarm occupancy LaTeX table")
    parser.add_argument(
        "--d-safe-cm", type=float, nargs="+", default=list(D_SAFE_TABLE_CM),
        help="Safety buffer centre-to-centre distances [cm]",
    )
    parser.add_argument(
        "--swarm-sizes", type=int, nargs="+", default=None,
        help="Swarm sizes N_b (default: paper set + N_max per D_safe)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write .tex file (prints to stdout if omitted)",
    )
    parser.add_argument(
        "--highlight", type=int, nargs="*", default=sorted(HIGHLIGHT_SWARM_SIZES),
        help="Rows to highlight (bold + blue background)",
    )
    args = parser.parse_args()

    sizes = args.swarm_sizes or default_swarm_sizes()
    tex = build_latex(args.d_safe_cm, sizes, highlight=set(args.highlight))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(tex)


if __name__ == "__main__":
    main()
