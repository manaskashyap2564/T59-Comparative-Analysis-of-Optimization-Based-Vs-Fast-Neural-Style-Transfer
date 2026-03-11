"""
Fast NST Inference — StyleSense (Fixed v2)
Returns runtime + content/style/tv losses for fair comparison.
Owner: Shubhansh Gupta
"""

import os, sys, time
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "../extractor"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../nst_optimization"))

from generator       import FastNSTGenerator
from load_checkpoint import load_extractor
from losses          import NSTLoss

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def normalize(x, device):
    return (x - MEAN.to(device)) / STD.to(device)

def load_image(path, size, device):
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

def save_image(tensor, path):
    img = tensor.squeeze(0).cpu().detach()
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    TF.to_pil_image(img).save(path)
    print(f"  Saved: {path}")


def run_fast_nst(
    content_path,
    style_path       = None,
    checkpoint       = "../../checkpoints/fast_nst_best.pth",
    extractor_ckpt   = "../../checkpoints/best_extractor.pth",
    output_path      = "../../outputs/fast_nst_result.jpg",
    img_size         = 256,
    device           = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load Generator ─────────────────────────────────────────
    generator = FastNSTGenerator(n_residual=5).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    generator.load_state_dict(ckpt["model_state_dict"])
    generator.eval()
    print(f"  Generator loaded from : {checkpoint}")

    # ── Load content image ─────────────────────────────────────
    content     = load_image(content_path, img_size, device)
    content_norm = normalize(content, device)

    # ── Warmup (exclude model-load from timing) ────────────────
    with torch.no_grad():
        _ = generator(content_norm)

    # ── Actual inference timing ────────────────────────────────
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output = generator(content_norm)
    if device == "cuda":
        torch.cuda.synchronize()
    runtime_ms = (time.time() - t0) * 1000

    save_image(output, output_path)
    print(f"  Runtime (actual)      : {runtime_ms:.2f} ms")

    # ── Compute losses for fair comparison ─────────────────────
    content_loss_val = style_loss_val = tv_loss_val = total_loss_val = 0.0

    if style_path is not None:
        extractor, _ = load_extractor(extractor_ckpt, device=device)
        extractor = extractor.to(device).eval()

        style_img   = load_image(style_path, img_size, device)
        style_norm  = normalize(style_img, device)

        with torch.no_grad():
            gen_norm    = normalize((output + 1.0) / 2.0, device)
            gen_feats   = extractor.get_feature_maps(gen_norm)
            cont_feats  = extractor.get_feature_maps(content_norm)
            style_feats = extractor.get_feature_maps(style_norm)

        criterion = NSTLoss(content_weight=1.0, style_weight=50.0, tv_weight=1e-4)
        with torch.no_grad():
            total, lc, ls, lt = criterion(
                output, gen_feats, cont_feats, style_feats)

        content_loss_val = lc.item()
        style_loss_val   = ls.item()
        tv_loss_val      = lt.item()
        total_loss_val   = total.item()

        print(f"  Content Loss          : {content_loss_val:.6f}")
        print(f"  Style Loss            : {style_loss_val:.6f}")
        print(f"  TV Loss               : {tv_loss_val:.6f}")
        print(f"  Total Loss            : {total_loss_val:.4f}")
    else:
        print("  [Note] style_path not given — losses skipped")

    return {
        "output_path"  : output_path,
        "runtime_ms"   : runtime_ms,
        "content_loss" : content_loss_val,
        "style_loss"   : style_loss_val,
        "tv_loss"      : tv_loss_val,
        "total_loss"   : total_loss_val,
    }


if __name__ == "__main__":
    import numpy as np

    os.makedirs("../../outputs/test_imgs", exist_ok=True)
    arr = np.zeros((256,256,3), dtype=np.uint8)
    for i in range(256):
        arr[i,:] = [i//2, i//3, 200]
    Image.fromarray(arr).save("../../outputs/test_imgs/content.jpg")

    result = run_fast_nst(
        content_path = "../../outputs/test_imgs/content.jpg",
        style_path   = "../../outputs/test_imgs/vangogh_style.jpg",
        output_path  = "../../outputs/test_imgs/fast_result.jpg",
        img_size     = 256,
    )
    print(f"\n  Result: {result}")
