"""
Benchmark Plots — StyleSense
Reads benchmark CSV and generates comparison charts.
Owner: Manas Kashyap
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


LOG_FILE  = "../../logs/benchmark_results.csv"
PLOTS_DIR = "../../logs/plots"


def read_csv(path):
    if not os.path.exists(path):
        print(f"  [!] CSV not found: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_runtime_comparison(rows, save_dir):
    """Bar chart — runtime comparison by method."""
    methods = defaultdict(list)
    for r in rows:
        methods[r["method"]].append(float(r["runtime_ms"]))

    names = list(methods.keys())
    avgs  = [sum(v)/len(v) for v in methods.values()]
    colors = ["#4C72B0", "#DD8452", "#55A868"][:len(names)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, avgs, color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.0fms", padding=5, fontsize=10)
    ax.set_title("Average Runtime: Optimization vs Fast NST",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Runtime (ms)", fontsize=11)
    ax.set_xlabel("Method", fontsize=11)
    ax.set_ylim(0, max(avgs) * 1.25)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(save_dir, "runtime_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_loss_curves(rows, save_dir):
    """Line chart — loss by iteration/run."""
    methods = defaultdict(lambda: {"total":[], "content":[], "style":[]})
    for r in rows:
        m = r["method"]
        methods[m]["total"].append(float(r["total_loss"]))
        methods[m]["content"].append(float(r["content_loss"]))
        methods[m]["style"].append(float(r["style_loss"]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    loss_types = ["total", "content", "style"]
    titles     = ["Total Loss", "Content Loss", "Style Loss"]
    colors     = {"optimization": "#4C72B0", "fast": "#DD8452"}

    for ax, lt, title in zip(axes, loss_types, titles):
        for method, data in methods.items():
            vals = data[lt]
            if vals:
                ax.plot(vals, label=method,
                        color=colors.get(method, "gray"),
                        linewidth=2, marker="o", markersize=4)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Run #", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("Loss Comparison: Optimization vs Fast NST",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "loss_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_quality_vs_speed(rows, save_dir):
    """Scatter — quality (content loss) vs speed (runtime)."""
    colors = {"optimization": "#4C72B0", "fast": "#DD8452"}
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in rows:
        ax.scatter(float(r["runtime_ms"]),
                   float(r["content_loss"]),
                   color=colors.get(r["method"], "gray"),
                   s=80, alpha=0.8, edgecolors="white", linewidth=0.5)

    patches = [mpatches.Patch(color=c, label=m)
               for m, c in colors.items()]
    ax.legend(handles=patches, fontsize=10)
    ax.set_title("Quality vs Speed Trade-off",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Runtime (ms) — lower is faster", fontsize=11)
    ax.set_ylabel("Content Loss — lower is better", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(save_dir, "quality_vs_speed.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def generate_all_plots():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    rows = read_csv(LOG_FILE)
    if not rows:
        print("  No data yet — run benchmark first.")
        return

    print(f" Generating plots from {len(rows)} runs...")
    plot_runtime_comparison(rows, PLOTS_DIR)
    plot_loss_curves(rows, PLOTS_DIR)
    plot_quality_vs_speed(rows, PLOTS_DIR)
    print(f"All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    # Add dummy data if CSV empty (for testing)
    os.makedirs("../../logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        import csv as _csv, time as _time
        with open(LOG_FILE, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=[
                "run_id","timestamp","method","content","style",
                "resolution","iterations","runtime_ms",
                "total_loss","content_loss","style_loss","tv_loss"])
            w.writeheader()
            for _ in range(5):
                w.writerow({"run_id":f"r{_}","timestamp":"2026-03-11",
                    "method":"optimization","content":"c.jpg","style":"s.jpg",
                    "resolution":256,"iterations":300,
                    "runtime_ms":4200+_*100,"total_loss":2.1-_*0.1,
                    "content_loss":0.05-_*0.005,"style_loss":0.8-_*0.05,
                    "tv_loss":0.01})
            for _ in range(5):
                w.writerow({"run_id":f"f{_}","timestamp":"2026-03-11",
                    "method":"fast","content":"c.jpg","style":"s.jpg",
                    "resolution":256,"iterations":1,
                    "runtime_ms":120+_*10,"total_loss":2.8-_*0.1,
                    "content_loss":0.08-_*0.005,"style_loss":1.1-_*0.08,
                    "tv_loss":0.02})

    generate_all_plots()
