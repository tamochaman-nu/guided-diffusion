"""
TEMPORARY DEBUG CODE (ACT-MONITOR): remove after diagnosis.

Analyze <logdir>/act_monitor/act.jsonl written by guided_diffusion/_debug_act_monitor.py.

    python scripts/_debug_act_monitor_analyze.py <logdir or act_monitor dir or act.jsonl> [--top 10]

Uses only the standard library; a per-level PNG is written if matplotlib is available.
"""

import argparse
import json
import math
import os
from collections import defaultdict

FP16_MAX = 65504.0


def load(path):
    if os.path.isdir(path):
        sub = os.path.join(path, "act_monitor")
        path = os.path.join(sub if os.path.isdir(sub) else path, "act.jsonl")
    records, events = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            (events if r.get("event") else records).append(r)
    return path, records, events


def slope(points):
    """Least-squares slope of log(max_abs) per 1000 steps."""
    pts = [(s, math.log(v)) for s, v in points if v is not None and math.isfinite(v) and v > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(s for s, _ in pts) / n
    my = sum(y for _, y in pts) / n
    var = sum((s - mx) ** 2 for s, _ in pts)
    if var == 0:
        return None
    return 1000.0 * sum((s - mx) * (y - my) for s, y in pts) / var


def fmt(v):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    path, records, events = load(args.path)
    if not records:
        print(f"no records in {path}")
        return

    series = defaultdict(list)  # module -> [(step, rec)]
    for r in records:
        series[r["module"]].append((r["step"], r))
    last_step = max(r["step"] for r in records)
    last = {r["module"]: r for r in records if r["step"] == last_step}
    steps = sorted({r["step"] for r in records})
    print(f"file: {path}")
    print(f"steps recorded: {len(steps)} ({steps[0]} .. {last_step}), modules: {len(series)}")

    # 1. largest max_abs at the last recorded step
    print(f"\n[1] top {args.top} max_abs at step {last_step} (fp16 limit {FP16_MAX:.0f})")
    rows = sorted(
        last.values(),
        key=lambda r: (r["max_abs"] if math.isfinite(r["max_abs"]) else math.inf),
        reverse=True,
    )
    for r in rows[: args.top]:
        print(
            f"  {r['module']:42s} {r['kind']:12s} L{r['level']} max_abs={fmt(r['max_abs']):>10s} "
            f"({r['max_abs'] / FP16_MAX:6.1%} of fp16) rms={fmt(r['rms'])} w_norm={fmt(r['w_norm'])}"
            if math.isfinite(r["max_abs"])
            else f"  {r['module']:42s} {r['kind']:12s} L{r['level']} max_abs=NONFINITE"
        )

    # 2. fastest growth of log(max_abs)
    slopes = {}
    for mod, pts in series.items():
        sl = slope([(s, r["max_abs"]) for s, r in pts])
        if sl is not None:
            slopes[mod] = sl
    print(f"\n[2] top {args.top} growth of log(max_abs) per 1000 steps (x factor per 1000 steps)")
    for mod, sl in sorted(slopes.items(), key=lambda kv: kv[1], reverse=True)[: args.top]:
        r = last[mod] if mod in last else series[mod][-1][1]
        wsl = slope([(s, x["w_norm"]) for s, x in series[mod]])
        extra = ""
        if "w_norm_emb" in r:
            extra = f" emb_w_norm slope={fmt(slope([(s, x['w_norm_emb']) for s, x in series[mod]]))}"
        print(
            f"  {mod:42s} {r['kind']:12s} L{r['level']} slope={sl:+.3f} (x{math.exp(min(sl, 700)):.3g}) "
            f"w_norm slope={fmt(wsl)}{extra}"
        )

    # 3. residual branch vs skip in each ResBlock
    print("\n[3] residual branch vs skip (per ResBlock, at last step and growth per 1000 steps)")
    n_branch = n_skip = 0
    rows = []
    for mod, r in last.items():
        if r["kind"] != "res_branch":
            continue
        base = mod[: -len(".out_layers")]
        skip = last.get(base + ".skip_connection")
        if skip is None:
            continue
        sb, ss = slopes.get(mod), slopes.get(base + ".skip_connection")
        dominant = "branch" if r["max_abs"] >= skip["max_abs"] else "skip"
        if dominant == "branch":
            n_branch += 1
        else:
            n_skip += 1
        rows.append((max(r["max_abs"], skip["max_abs"]), base, r, skip, sb, ss, dominant))
    rows.sort(key=lambda x: x[0] if math.isfinite(x[0]) else math.inf, reverse=True)
    print(f"  larger at last step: branch in {n_branch} blocks, skip in {n_skip} blocks")
    for _, base, r, skip, sb, ss, dominant in rows[: args.top]:
        print(
            f"  {base:36s} L{r['level']} branch={fmt(r['max_abs']):>10s} (slope {fmt(sb)}) "
            f"skip={fmt(skip['max_abs']):>10s} (slope {fmt(ss)}) -> {dominant}"
        )

    # 4. first non-finite activation
    print("\n[4] first non-finite activation")
    if events:
        e = min(events, key=lambda e: e["step"])
        print(f"  step {e['step']}: {e['module']} ({e['kind']})")
        for p in e.get("preceding", []):
            print(f"    preceded by {p['module']:40s} max_abs={fmt(p.get('max_abs'))} rms={fmt(p.get('rms'))}")
    else:
        bad = [r for r in records if r["nonfinite"]]
        if bad:
            b = min(bad, key=lambda r: r["step"])
            print(f"  (from records) step {b['step']}: {b['module']}")
        else:
            print("  none")

    # 5. (optional) max_abs per resolution level
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[5] matplotlib not available, skipped plot")
        return
    by_level = defaultdict(lambda: defaultdict(float))
    for r in records:
        v = r["max_abs"] if math.isfinite(r["max_abs"]) else FP16_MAX * 2
        lv = by_level[r["level"]]
        lv[r["step"]] = max(lv[r["step"]], v)
    fig, ax = plt.subplots(figsize=(8, 5))
    for level in sorted(by_level):
        pts = sorted(by_level[level].items())
        ax.plot([s for s, _ in pts], [v for _, v in pts], label=f"level {level}")
    ax.axhline(FP16_MAX, color="red", linestyle="--", label="fp16 max")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("max |activation|")
    ax.legend()
    out = os.path.join(os.path.dirname(path), "max_abs_by_level.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n[5] plot: {out}")


if __name__ == "__main__":
    main()
