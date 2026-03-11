"""
Full Benchmark Runner — StyleSense
Runs both NST methods on test pairs, logs CSV, generates plots.
Owner: Manas Kashyap
"""

import os, sys, csv, time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "../comparison"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../nst_optimization"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../nst_fast"))

from compare       import run_comparison
from plot_metrics  import generate_all_plots

LOG_FILE   = "../../logs/benchmark_results.csv"
FIELDNAMES = ["run_id","timestamp","method","content","style",
              "resolution","runtime_ms","content_loss",
              "style_loss","tv_loss","total_loss"]


def init_log():
    os.makedirs("../../logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def log_result(method, content, style, resolution, metrics):
    init_log()
    row = {
        "run_id"      : f"run_{int(time.time())}_{method[:3]}",
        "timestamp"   : datetime.now().isoformat(),
        "method"      : method,
        "content"     : os.path.basename(content),
        "style"       : os.path.basename(style),
        "resolution"  : resolution,
        "runtime_ms"  : round(metrics.get("runtime_ms",   0), 2),
        "content_loss": round(metrics.get("content_loss", 0), 6),
        "style_loss"  : round(metrics.get("style_loss",   0), 6),
        "tv_loss"     : round(metrics.get("tv_loss",      0), 6),
        "total_loss"  : round(metrics.get("total_loss",   0), 6),
    }
    with open(LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
    print(f"  Logged: {row['run_id']} | {method:<14} | "
          f"{resolution}px | {row['runtime_ms']:.1f}ms")
    return row


def run_full_benchmark(
    test_pairs,
    resolutions      = [128, 256],
    opt_iterations   = 200,
    opt_checkpoint   = "../../checkpoints/best_extractor.pth",
    fast_checkpoint  = "../../checkpoints/fast_nst_best.pth",
):
    print("\n" + "=" * 60)
    print("  StyleSense — Full Benchmark")
    print(f"  Pairs       : {len(test_pairs)}")
    print(f"  Resolutions : {resolutions}")
    print(f"  Opt iters   : {opt_iterations}")
    print("=" * 60)

    for c_path, s_path in test_pairs:
        for res in resolutions:
            results = run_comparison(
                content_path  = c_path,
                style_path    = s_path,
                img_size      = res,
                opt_iterations= opt_iterations,
                opt_checkpoint = opt_checkpoint,
                fast_checkpoint= fast_checkpoint,
                output_dir    = f"../../outputs/benchmark_{res}px",
            )
            for method, metrics in results.items():
                log_result(method, c_path, s_path, res, metrics)

    print("\n  Benchmark complete! Generating plots...")
    generate_all_plots()
    print(f"  Results CSV : {LOG_FILE}")
    print(f"  Plots dir   : ../../logs/plots/")


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    os.makedirs("../../outputs/test_imgs", exist_ok=True)

    # Content image
    arr = np.zeros((256,256,3), dtype=np.uint8)
    for i in range(256):
        arr[i,:] = [i//2, i//3, 200]
    Image.fromarray(arr).save("../../outputs/test_imgs/content.jpg")

    run_full_benchmark(
        test_pairs  = [("../../outputs/test_imgs/content.jpg",
                         "../../outputs/test_imgs/vangogh_style.jpg")],
        resolutions = [128],
        opt_iterations = 50,
    )
