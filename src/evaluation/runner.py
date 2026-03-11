"""
Evaluation Runner — StyleSense
Runs NST on test pairs, collects metrics, saves CSV + summary.
Owner: Manas Kashyap
"""

import os, sys, csv, time, json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../nst_optimization"))
from optimizer_nst import run_optimization_nst


FIELDNAMES = ["run_id","timestamp","method","content","style",
              "resolution","iterations","runtime_ms",
              "total_loss","content_loss","style_loss","tv_loss"]

LOG_FILE = "../../logs/benchmark_results.csv"


def init_log():
    os.makedirs("../../logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def log_result(method, content, style, resolution, iterations, metrics):
    init_log()
    row = {
        "run_id"       : f"run_{int(time.time())}",
        "timestamp"    : datetime.now().isoformat(),
        "method"       : method,
        "content"      : os.path.basename(content),
        "style"        : os.path.basename(style),
        "resolution"   : resolution,
        "iterations"   : iterations,
        "runtime_ms"   : round(metrics.get("runtime_ms", 0), 2),
        "total_loss"   : round(metrics.get("total_loss", 0), 6),
        "content_loss" : round(metrics.get("content_loss", 0), 6),
        "style_loss"   : round(metrics.get("style_loss", 0), 6),
        "tv_loss"      : round(metrics.get("tv_loss", 0), 6),
    }
    with open(LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
    print(f"  Logged: {row['run_id']} | {method} | {resolution}px | "
          f"{row['runtime_ms']:.0f}ms")
    return row


def run_benchmark(test_pairs: list, resolutions=[128, 256],
                  iterations=100, checkpoint="../../checkpoints/best_extractor.pth"):
    """
    Runs optimization-based NST on all test_pairs x resolutions.
    test_pairs: list of (content_path, style_path)
    """
    print("=" * 55)
    print("  StyleSense — Benchmark Runner")
    print(f"  Pairs: {len(test_pairs)} | Resolutions: {resolutions}")
    print("=" * 55)

    for c_path, s_path in test_pairs:
        for res in resolutions:
            out = f"../../outputs/bench_{os.path.basename(c_path).split('.')[0]}"                   f"_{res}px.jpg"
            metrics = run_optimization_nst(
                content_path  = c_path,
                style_path    = s_path,
                checkpoint    = checkpoint,
                output_path   = out,
                img_size      = res,
                iterations    = iterations,
                save_every    = iterations,
            )
            log_result("optimization", c_path, s_path, res, iterations, metrics)

    print(f"\n  Benchmark complete. Results in: {LOG_FILE}")


if __name__ == "__main__":
    # Quick smoke test using dummy images
    import numpy as np
    from PIL import Image
    os.makedirs("../../outputs/test_imgs", exist_ok=True)

    for name in ["content.jpg", "style.jpg"]:
        img = Image.fromarray(
            np.random.randint(50,200,(256,256,3),dtype=np.uint8))
        img.save(f"../../outputs/test_imgs/{name}")

    run_benchmark(
        test_pairs  = [("../../outputs/test_imgs/content.jpg",
                         "../../outputs/test_imgs/style.jpg")],
        resolutions = [128],
        iterations  = 20,
    )
