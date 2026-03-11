"""
Side-by-Side Comparison — Optimization NST vs Fast NST
Runs both methods on same content+style, logs metrics.
Owner: Shubhansh Gupta
"""

import os, sys, time
import torch
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../nst_optimization"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../nst_fast"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../extractor"))

from optimizer_nst import run_optimization_nst
from inference     import run_fast_nst


def run_comparison(
    content_path,
    style_path,
    img_size         = 256,
    opt_iterations   = 300,
    opt_checkpoint   = "../../checkpoints/best_extractor.pth",
    fast_checkpoint  = "../../checkpoints/fast_nst_best.pth",
    output_dir       = "../../outputs/comparison",
    device           = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 55)
    print("  StyleSense — Method Comparison")
    print("=" * 55)
    print(f"  Content : {os.path.basename(content_path)}")
    print(f"  Style   : {os.path.basename(style_path)}")
    print(f"  Size    : {img_size}x{img_size}")
    print("=" * 55)

    results = {}

    # ── Fast NST ──────────────────────────────────────────────
    print("\n  [1/2] Running Fast NST...")
    fast_out = os.path.join(output_dir, "fast_result.jpg")
    fast_metrics = run_fast_nst(
        content_path  = content_path,
        style_path    = style_path,          # ← yeh line ADD karo
        checkpoint    = fast_checkpoint,
        extractor_ckpt= opt_checkpoint,      # ← yeh line ADD karo
        output_path   = fast_out,
        img_size      = img_size,
        device        = device
        # content_path = content_path,
        # checkpoint   = fast_checkpoint,
        # output_path  = fast_out,
        # img_size     = img_size,
        # device       = device
    )
    results["fast"] = fast_metrics
    print(f"  Fast NST done → {fast_metrics['runtime_ms']:.1f} ms")

    # ── Optimization NST ──────────────────────────────────────
    print("\n  [2/2] Running Optimization NST...")
    opt_out = os.path.join(output_dir, "opt_result.jpg")
    opt_metrics = run_optimization_nst(
        content_path = content_path,
        style_path   = style_path,
        checkpoint   = opt_checkpoint,
        output_path  = opt_out,
        img_size     = img_size,
        iterations   = opt_iterations,
        save_every   = opt_iterations,
        device       = device
    )
    results["optimization"] = opt_metrics
    print(f"  Opt NST done  → {opt_metrics['runtime_ms']:.1f} ms")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  COMPARISON SUMMARY")
    print("=" * 55)
    print(f"  {'Method':<18} {'Runtime':>12} {'Content Loss':>14} {'Style Loss':>12}")
    print(f"  {'-'*55}")
    for method, m in results.items():
        print(f"  {method:<18} {m['runtime_ms']:>10.1f}ms "
              f"{m.get('content_loss',0):>14.6f} "
              f"{m.get('style_loss',0):>12.6f}")

    # speedup = results["optimization"]["runtime_ms"] / results["fast"]["runtime_ms"]
    fast_rt = max(results["fast"]["runtime_ms"], 0.1)   # zero se bachao
    speedup = results["optimization"]["runtime_ms"] / fast_rt
    print(f"\n  ⚡ Fast NST is {speedup:.0f}x faster than Optimization NST")
    print("=" * 55)

    # ── Side-by-side image ────────────────────────────────────
    try:
        size = (img_size, img_size)
        imgs = [
            Image.open(content_path).resize(size),
            Image.open(style_path).resize(size),
            Image.open(fast_out).resize(size),
            Image.open(opt_out).resize(size),
        ]
        labels  = ["Content", "Style", "Fast NST", "Opt NST"]
        w, h    = img_size, img_size
        canvas  = Image.new("RGB", (w * 4 + 30, h + 40), (245, 245, 245))
        for i, (img, label) in enumerate(zip(imgs, labels)):
            canvas.paste(img, (i * (w + 10), 40))
        sidebyside = os.path.join(output_dir, "side_by_side.jpg")
        canvas.save(sidebyside)
        print(f"\n  Side-by-side saved: {sidebyside}")
    except Exception as e:
        print(f"  [Note] Side-by-side skipped: {e}")

    return results


if __name__ == "__main__":
    import numpy as np

    os.makedirs("../../outputs/test_imgs", exist_ok=True)

    # Content image
    arr = np.zeros((256,256,3), dtype=np.uint8)
    for i in range(256):
        arr[i,:] = [i//2, i//3, 200]
    Image.fromarray(arr).save("../../outputs/test_imgs/content.jpg")

    run_comparison(
        content_path  = "../../outputs/test_imgs/content.jpg",
        style_path    = "../../outputs/test_imgs/vangogh_style.jpg",
        img_size      = 128,
        opt_iterations= 100,
    )
