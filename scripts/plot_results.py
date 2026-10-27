#!/usr/bin/env python3
"""Plot gpumem CSV results.

Writes <out-dir>/bandwidth_vs_size.png (read/write/copy) and <out-dir>/latency_vs_size.png
(pointer chase: cycles per load, then ns per load). The x-axis is log2 buffer size, with
dashed reference lines at the L2 size from each CSV header plus any --ref you pass.

Examples
  scripts/plot_results.py                                   # newest run in results/
  scripts/plot_results.py results/cuda_*_20261027-*.csv
  scripts/plot_results.py run_a/*.csv run_b/*.csv --label "RTX 5090" --label "M3 Max"
  scripts/plot_results.py --ref L1=128K --ref "L1 (x170 SMs)=21760K"

One "run" is the set of CSVs sharing a backend + timestamp (one `gpumem --test all`).
Give one --label per run, in the order the runs first appear on the command line.
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

# Validated categorical palette (fixed order, never cycled) and light-surface ink tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
CRITICAL = "#d03b3b"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

STREAM_TESTS = ["read", "write", "copy"]
MAX_DIRECT_LABELS = 4


# ---------------------------------------------------------------------------------------------
# Loading


def parse_size(text):
    m = re.fullmatch(r"(\d+)\s*([kmg]?)(i?b)?", text.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError(f"invalid size '{text}' (e.g. 128K, 96M, 1G)")
    return int(m.group(1)) * {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[m.group(2)]


def format_size(n, short=False):
    for unit, scale in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if n >= scale:
            value = f"{n / scale:g}"
            return f"{value}{unit}" if short else f"{value} {unit}iB"
    return f"{n:g}" if short else f"{n:g} B"


def read_result(path):
    meta, body = {}, []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("# "):
                key, _, value = line[2:].rstrip("\r\n").partition(": ")
                meta[key] = value
            else:
                body.append(line)
    return meta, list(csv.DictReader(body))


def default_inputs(results_dir):
    """CSVs of the newest run directly in results_dir (subdirectories such as smoke/ are skipped)."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not paths:
        return []
    stamps = {p: read_result(p)[0].get("timestamp", "") for p in paths}
    newest = max(stamps.values())
    return [p for p in paths if stamps[p] == newest]


def load_runs(paths, labels):
    runs = OrderedDict()
    for path in paths:
        meta, rows = read_result(path)
        if meta.get("tool") != "gpumem":
            sys.exit(f"{path}: not a gpumem CSV (missing '# tool: gpumem')")
        key = (meta.get("backend", "?"), meta.get("timestamp", "?"), meta.get("device", "?"))
        run = runs.setdefault(key, {"meta": meta, "tests": {}})
        run["tests"][meta.get("test", "?")] = rows

    if len(runs) > len(SERIES):
        sys.exit(f"{len(runs)} runs given; at most {len(SERIES)} can be compared in one plot")
    if labels and len(labels) != len(runs):
        found = "\n  ".join(f"{b} {t} ({d})" for b, t, d in runs)
        sys.exit(f"got {len(labels)} --label for {len(runs)} runs:\n  {found}")

    out = []
    for i, ((backend, stamp, device), run) in enumerate(runs.items()):
        label = labels[i] if labels else (device if len(runs) == 1 else f"{device} [{backend} {stamp}]")
        out.append({"label": label, "meta": run["meta"], "tests": run["tests"]})
    return out


def series_points(rows, y_key):
    pts = []
    for r in rows:
        if r.get(y_key, "") == "":
            continue
        pts.append((int(r["size_bytes"]), float(r[y_key]), r.get("check") == "PASS"))
    return pts


# ---------------------------------------------------------------------------------------------
# Drawing


def style_axes(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_2, labelsize=9)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)


def set_size_ticks(ax, sizes):
    lo, hi = min(sizes), max(sizes)
    ticks = [1 << p for p in range(64) if lo <= (1 << p) <= hi]
    if len(ticks) > 12:
        ticks = ticks[::2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: format_size(int(x), short=True)))
    ax.set_xlim(lo / 1.25, hi * 1.25)


def draw_refs(ax, refs, label=True):
    """Dashed reference lines; labels sit above the plot area (staggered) so they never cover data."""
    for i, (size, text) in enumerate(sorted(refs)):
        ax.axvline(size, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        if label:
            ax.annotate(text, (size, 1.0), xycoords=ax.get_xaxis_transform(), xytext=(0, 4 + 12 * (i % 2)),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=MUTED,
                        annotation_clip=False)


def draw_series(ax, series):
    """series: list of (name, points, slot). Returns (any_failed, line ends, max y)."""
    failed_any = False
    ends = []
    ymax = 0.0
    for name, pts, slot in series:
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=SERIES[slot], linewidth=2, marker=MARKERS[slot], markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=name, zorder=3)
        bad = [(x, y) for x, y, ok in pts if not ok]
        if bad:
            failed_any = True
            ax.scatter([b[0] for b in bad], [b[1] for b in bad], marker="x", s=60, linewidths=2,
                       color=CRITICAL, zorder=4)
        ends.append((name, xs[-1], ys[-1]))
        ymax = max(ymax, max(ys))
    return failed_any, ends, ymax


def finalize_axes(axes, ends_per_axis, ymaxes, shared_y):
    """Zero-based y limits (shared across panels when shared_y), then direct end labels.

    Limits are set only after all data is drawn: set_ylim disables y-autoscaling, and label
    placement needs the final limits.
    """
    for ax, ymax in zip(axes, ymaxes):
        top = max(ymaxes) if shared_y else ymax
        ax.set_ylim(bottom=0, top=(top or 1) * 1.08)
    for ax, ends in zip(axes, ends_per_axis):
        if 1 < len(ends) <= MAX_DIRECT_LABELS:
            place_end_labels(ax, ends)


def place_end_labels(ax, ends, min_gap_px=12):
    """Direct labels at line ends, nudged apart vertically so they never overlap."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData
    items = sorted(((to_px.transform((x, y))[1], name, x, y) for name, x, y in ends))
    placed = []
    for py, name, x, y in items:
        if placed and py - placed[-1][0] < min_gap_px:
            py = placed[-1][0] + min_gap_px
        placed.append((py, name, x, y))
    for py, name, x, y in placed:
        _, natural = to_px.transform((x, y))
        ax.annotate(name, (x, y), xytext=(8, py - natural), textcoords="offset pixels",
                    va="center", ha="left", fontsize=8.5, color=TEXT_2, annotation_clip=False)


def finish(fig, axes, title, failed, path, has_legend):
    handles, names = [], []
    for ax in axes:
        for h, n in zip(*ax.get_legend_handles_labels()):
            if n not in names:
                handles.append(h)
                names.append(n)
    if failed:
        handles.append(plt.Line2D([], [], marker="x", color=CRITICAL, linestyle="", markersize=8,
                                  markeredgewidth=2))
        names.append("check FAILED (value not trustworthy)")
    if has_legend or failed:
        fig.legend(handles, names, loc="upper center", ncol=min(len(names), 5), frameon=False,
                   fontsize=9, labelcolor=TEXT_2, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.5, y=0.995)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def ref_lines(runs, extra_refs, include_l2):
    refs = []
    if include_l2:
        l2s = OrderedDict()
        for run in runs:
            l2 = int(run["meta"].get("l2_bytes", "0") or 0)
            if l2:
                l2s.setdefault(l2, []).append(run["label"])
        for l2, owners in l2s.items():
            who = "" if len(l2s) == 1 else f" ({', '.join(owners)})"
            refs.append((l2, f"L2 {format_size(l2)}{who}"))
    refs.extend((size, f"{name} {format_size(size)}") for name, size in extra_refs)
    return refs


def plot_bandwidth(runs, refs, out_path, title):
    present = [t for t in STREAM_TESTS if any(t in r["tests"] for r in runs)]
    if not present:
        print("no read/write/copy results; skipping bandwidth plot")
        return
    sizes = [int(row["size_bytes"]) for r in runs for t in present for row in r["tests"].get(t, [])]

    if len(runs) == 1:
        # One run: one panel, one series per test.
        fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=SURFACE)
        panels = [(ax, None, [(t, series_points(runs[0]["tests"].get(t, []), "bandwidth_gbs"),
                               STREAM_TESTS.index(t)) for t in present])]
        n_series = len(present)
        fig.subplots_adjust(top=0.78, bottom=0.11, left=0.08, right=0.91)
    else:
        # Several runs: small multiples by test, one series per run, shared y.
        fig, grid = plt.subplots(1, len(present), figsize=(5.6 * len(present), 6.0), sharey=True,
                                 facecolor=SURFACE, squeeze=False)
        panels = [(ax, t, [(r["label"], series_points(r["tests"].get(t, []), "bandwidth_gbs"), i)
                           for i, r in enumerate(runs)]) for ax, t in zip(grid[0], present)]
        n_series = len(runs)
        fig.subplots_adjust(top=0.76, bottom=0.11, left=0.06, right=0.95, wspace=0.22)

    failed, all_ends, ymaxes = False, [], []
    for ax, test, series in panels:
        style_axes(ax, "Bandwidth (GB/s)" if ax is panels[0][0] else "")
        set_size_ticks(ax, sizes)
        draw_refs(ax, refs)
        bad, ends, ymax = draw_series(ax, series)
        failed |= bad
        all_ends.append(ends)
        ymaxes.append(ymax)
        if test:
            ax.set_title(test, color=TEXT, fontsize=10, loc="left", pad=34 if refs else 6)
        ax.set_xlabel("Buffer size", color=TEXT_2)
    axes = [p[0] for p in panels]
    finalize_axes(axes, all_ends, ymaxes, shared_y=True)
    finish(fig, axes, title or "Memory bandwidth vs buffer size", failed, out_path, n_series > 1)


def plot_latency(runs, refs, out_path, title):
    chase_runs = [(i, r) for i, r in enumerate(runs) if "chase" in r["tests"]]
    if not chase_runs:
        print("no chase results; skipping latency plot")
        return
    has_cycles = any(row.get("cycles_per_load") for _, r in chase_runs for row in r["tests"]["chase"])
    panels = [("cycles_per_load", "Cycles per load")] if has_cycles else []
    panels.append(("ns_per_load", "ns per load"))
    sizes = [int(row["size_bytes"]) for _, r in chase_runs for row in r["tests"]["chase"]]

    # Two measures on different scales -> stacked panels sharing x, never a second y-axis.
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.6 * len(panels) + 1), sharex=True,
                             facecolor=SURFACE, squeeze=False)
    axes = list(axes[:, 0])
    top = 0.84 if len(panels) == 2 else 0.74
    fig.subplots_adjust(top=top, bottom=0.08, left=0.08, right=0.89, hspace=0.12)

    failed, all_ends, ymaxes = False, [], []
    for ax, (key, ylabel) in zip(axes, panels):
        style_axes(ax, ylabel)
        set_size_ticks(ax, sizes)
        draw_refs(ax, refs, label=ax is axes[0])
        series = [(r["label"], series_points(r["tests"]["chase"], key), i) for i, r in chase_runs]
        bad, ends, ymax = draw_series(ax, series)
        failed |= bad
        all_ends.append(ends)
        ymaxes.append(ymax)
    axes[-1].set_xlabel("Buffer size (pointer-chase working set)", color=TEXT_2)
    finalize_axes(axes, all_ends, ymaxes, shared_y=False)
    finish(fig, axes, title or "Load-to-use latency vs working-set size", failed, out_path, len(chase_runs) > 1)


# ---------------------------------------------------------------------------------------------


def parse_ref(text):
    name, sep, size = text.rpartition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"--ref expects NAME=SIZE, got '{text}'")
    return name, parse_size(size)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="*", help="gpumem CSV files (default: newest run in results/)")
    p.add_argument("--label", action="append", default=[], help="legend name per run, in order (repeatable)")
    p.add_argument("--out-dir", default="results", help="where to write the PNGs (default: results)")
    p.add_argument("--ref", action="append", type=parse_ref, default=[], metavar="NAME=SIZE",
                   help="extra reference line, e.g. L1=128K (repeatable)")
    p.add_argument("--no-l2", action="store_true", help="omit the L2 reference line from the CSV header")
    p.add_argument("--title", help="override the figure titles' prefix")
    args = p.parse_args(argv)

    paths = args.csv or default_inputs("results")
    if not paths:
        sys.exit("no CSVs given and none found in results/")
    runs = load_runs(paths, args.label)
    refs = ref_lines(runs, args.ref, not args.no_l2)

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = f"{args.title}: " if args.title else ""
    plot_bandwidth(runs, refs, os.path.join(args.out_dir, "bandwidth_vs_size.png"),
                   prefix + "Memory bandwidth vs buffer size" if prefix else None)
    plot_latency(runs, refs, os.path.join(args.out_dir, "latency_vs_size.png"),
                 prefix + "Load-to-use latency vs working-set size" if prefix else None)

    for run in runs:
        for test, rows in run["tests"].items():
            bad = [r for r in rows if r.get("check") != "PASS"]
            if bad:
                print(f"warning: {run['label']} {test}: {len(bad)} size(s) failed the correctness check",
                      file=sys.stderr)


if __name__ == "__main__":
    main()
