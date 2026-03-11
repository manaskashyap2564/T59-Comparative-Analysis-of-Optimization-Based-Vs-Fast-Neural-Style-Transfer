"""
Optimization-Based NST — StyleSense (Fixed)
Owner: Shubhansh Gupta
"""

import os, sys, time
import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../extractor"))
from vgg_like_cnn    import VGGLikeExtractor
from load_checkpoint import load_extractor
from losses          import NSTLoss

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def load_image(path, size=256, device="cpu"):
    img = Image.open(path).convert("RGB")
    tf  = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std =[0.229,0.224,0.225])
    ])
    return tf(img).unsqueeze(0).to(device)

def save_image(tensor, path):
    img = tensor.squeeze(0).cpu().detach()
    img = img * STD + MEAN
    img = img.clamp(0, 1)
    TF.to_pil_image(img).save(path)
    print(f"  Saved: {path}")

def run_optimization_nst(
    content_path,
    style_path,
    checkpoint    = "../../checkpoints/best_extractor.pth",
    output_path   = "../../outputs/opt_nst_result.jpg",
    img_size      = 256,
    iterations    = 300,
    content_weight= 1.0,
    style_weight  = 1e5,       # ← tuned down from 1e6
    tv_weight     = 1e-4,      # ← tuned down from 1e-3
    lr            = 0.05,      # ← increased from 0.02
    save_every    = 50,
    device        = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Device     : {device}")
    print(f"  Iterations : {iterations}")
    print(f"  Image size : {img_size}x{img_size}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # # Load frozen extractor
    # extractor, _ = load_extractor(checkpoint, device=device)
    # extractor = extractor.to(device).eval()
    extractor, _ = load_extractor(checkpoint, device=device)
    extractor = extractor.to(device).eval()

    # Load images
    content_img = load_image(content_path, img_size, device)
    style_img   = load_image(style_path,   img_size, device)
    
    # ← Add this debug line
    print(f"  Extractor device: {next(extractor.parameters()).device}")
    print(f"  Content img device: {content_img.device}")
    # print(f"  Gen img device: {gen_img.device}")

    # Precompute fixed features
    with torch.no_grad():
        content_features = extractor.get_feature_maps(content_img)
        style_features   = extractor.get_feature_maps(style_img)

    # Print loss magnitudes for debugging
    print(f"\n  [Debug] Content feat (block3) mean: "
          f"{content_features['block3'].abs().mean().item():.4f}")
    print(f"  [Debug] Style feat   (block3) mean: "
          f"{style_features['block3'].abs().mean().item():.4f}")

    # Start from content image
    gen_img = content_img.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([gen_img], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=100, gamma=0.5)
    criterion = NSTLoss(content_weight, style_weight, tv_weight)

    print(f"\n  {'Iter':<8}{'Total':>12}{'Content':>12}"
          f"{'Style':>12}{'TV':>10}{'Time':>8}")
    print(f"  {'-'*62}")

    t_start = time.time()
    for i in range(1, iterations + 1):
        optimizer.zero_grad()
        gen_features = extractor.get_feature_maps(gen_img)
        total, lc, ls, lt = criterion(
            gen_img, gen_features, content_features, style_features)
        total.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            gen_img.data.clamp_(-3.0, 3.0)

        if i % save_every == 0 or i == 1:
            elapsed = time.time() - t_start
            print(f"  {i:<8}{total.item():>12.4f}{lc.item():>12.6f}"
                  f"{ls.item():>12.6f}{lt.item():>10.6f}{elapsed:>7.1f}s")
            mid = output_path.replace(".jpg", f"_iter{i}.jpg")
            save_image(gen_img, mid)

    save_image(gen_img, output_path)
    rt = time.time() - t_start
    print(f"\n  ✅ Done! Runtime: {rt:.1f}s")

    return {"output_path": output_path,
            "total_loss":   total.item(),
            "content_loss": lc.item(),
            "style_loss":   ls.item(),
            "tv_loss":      lt.item(),
            "runtime_ms":   rt * 1000}


if __name__ == "__main__":
    # ── Create REAL-looking test images (not random noise!) ──
    os.makedirs("../../outputs/test_imgs", exist_ok=True)

    # Content: blue sky gradient
    content_arr = np.zeros((256,256,3), dtype=np.uint8)
    for i in range(256):
        content_arr[i,:] = [i//2, i//3, 200]   # gradient
    Image.fromarray(content_arr).save("../../outputs/test_imgs/content.jpg")

    # Style: warm orange pattern
    style_arr = np.zeros((256,256,3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            style_arr[i,j] = [200, (i+j)%256, 50]
    Image.fromarray(style_arr).save("../../outputs/test_imgs/style.jpg")

    result = run_optimization_nst(
    content_path  = "../../outputs/test_imgs/content.jpg",
    style_path    = "../../outputs/test_imgs/style.jpg",
    output_path   = "../../outputs/test_imgs/opt_result.jpg",
    img_size      = 128,
    iterations    = 300,
    save_every    = 100,    # 10 → 100 (sirf 3 saves)
    )

    # result = run_optimization_nst(
    #     content_path  = "../../outputs/test_imgs/content.jpg",
    #     style_path    = "../../outputs/test_imgs/style.jpg",
    #     output_path   = "../../outputs/test_imgs/opt_result.jpg",
    #     img_size      = 128,
    #     iterations    = 100,
    #     save_every    = 20,
    # )
    print("\n  Metrics:", result)
# """
# Optimization-Based NST — StyleSense
# Iteratively optimizes a generated image to match content + style.
# Owner: Shubhansh Gupta
# """

# import os
# import sys
# import time
# import torch
# import torch.optim as optim
# from PIL import Image
# import torchvision.transforms as T
# import torchvision.transforms.functional as TF

# sys.path.append(os.path.join(os.path.dirname(__file__), "../extractor"))
# from vgg_like_cnn    import VGGLikeExtractor
# from load_checkpoint import load_extractor
# from losses          import NSTLoss


# # ── Image Utils ────────────────────────────────────────────────────────────────
# MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
# STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# def load_image(path: str, size: int = 256, device="cpu") -> torch.Tensor:
#     img = Image.open(path).convert("RGB")
#     tf  = T.Compose([
#         T.Resize((size, size)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485,0.456,0.406],
#                     std =[0.229,0.224,0.225])
#     ])
#     return tf(img).unsqueeze(0).to(device)


# def save_image(tensor: torch.Tensor, path: str):
#     img = tensor.squeeze(0).cpu().detach()
#     img = img * STD + MEAN          # unnormalize
#     img = img.clamp(0, 1)
#     TF.to_pil_image(img).save(path)
#     print(f"  Saved: {path}")


# # ── Main NST Function ──────────────────────────────────────────────────────────
# def run_optimization_nst(
#     content_path : str,
#     style_path   : str,
#     checkpoint   : str = "../../checkpoints/best_extractor.pth",
#     output_path  : str = "../../outputs/opt_nst_result.jpg",
#     img_size     : int = 256,
#     iterations   : int = 500,
#     content_weight: float = 1.0,
#     style_weight  : float = 1e6,
#     tv_weight     : float = 1e-3,
#     lr            : float = 0.02,
#     save_every    : int = 100,
#     device        : str = None
# ):
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"  Device     : {device}")
#     print(f"  Iterations : {iterations}")
#     print(f"  Image size : {img_size}x{img_size}")

#     os.makedirs("../../outputs", exist_ok=True)

#     # Load extractor (frozen backbone)
#     extractor, _ = load_extractor(checkpoint, device=device)
#     extractor     = extractor.to(device).eval()

#     # Load images
#     content_img = load_image(content_path, img_size, device)
#     style_img   = load_image(style_path,   img_size, device)

#     # Precompute fixed content + style features
#     with torch.no_grad():
#         content_features = extractor.get_feature_maps(content_img)
#         style_features   = extractor.get_feature_maps(style_img)

#     # Generated image — start from content image (faster convergence)
#     # gen_img = content_img.clone().requires_grad_(True)
#     gen_img = content_img.clone().detach().to(device).requires_grad_(True)


#     optimizer = optim.Adam([gen_img], lr=lr)
#     criterion = NSTLoss(content_weight, style_weight, tv_weight)

#     print(f"\n  Starting optimization...")
#     print(f"  {'Iter':<8} {'Total':>12} {'Content':>12} {'Style':>12} {'TV':>10} {'Time':>8}")
#     print(f"  {'-'*60}")

#     t_start = time.time()
#     for i in range(1, iterations + 1):
#         optimizer.zero_grad()
#         gen_features = extractor.get_feature_maps(gen_img)
#         total, lc, ls, lt = criterion(gen_img, gen_features,
#                                        content_features, style_features)
#         total.backward()
#         optimizer.step()

#         # Clamp to valid range
#         with torch.no_grad():
#             gen_img.data.clamp_(-2.5, 2.5)

#         if i % save_every == 0 or i == 1:
#             elapsed = time.time() - t_start
#             print(f"  {i:<8} {total.item():>12.2f} {lc.item():>12.6f} "
#                   f"{ls.item():>12.6f} {lt.item():>10.6f} {elapsed:>7.1f}s")

#             # Save intermediate result
#             mid_path = output_path.replace(".jpg", f"_iter{i}.jpg")
#             save_image(gen_img, mid_path)

#     # Save final result
#     save_image(gen_img, output_path)
#     total_time = time.time() - t_start
#     print(f"\n  ✅ Done! Total time: {total_time:.1f}s")
#     print(f"  Final output: {output_path}")

#     return {
#         "output_path"  : output_path,
#         "total_loss"   : total.item(),
#         "content_loss" : lc.item(),
#         "style_loss"   : ls.item(),
#         "tv_loss"      : lt.item(),
#         "runtime_ms"   : total_time * 1000
#     }


# # ── Smoke Test (dummy images) ──────────────────────────────────────────────────
# if __name__ == "__main__":
#     import numpy as np
#     from PIL import Image as PILImage

#     # Create dummy content + style images for testing
#     os.makedirs("../../outputs/test_imgs", exist_ok=True)
#     dummy_c = PILImage.fromarray(
#         np.random.randint(50, 200, (256,256,3), dtype=np.uint8))
#     dummy_s = PILImage.fromarray(
#         np.random.randint(50, 200, (256,256,3), dtype=np.uint8))
#     dummy_c.save("../../outputs/test_imgs/content.jpg")
#     dummy_s.save("../../outputs/test_imgs/style.jpg")

#     result = run_optimization_nst(
#     content_path = "../../outputs/test_imgs/content.jpg",
#     style_path   = "../../outputs/test_imgs/style.jpg",
#     output_path  = "../../outputs/test_imgs/opt_result.jpg",
#     img_size     = 128,
#     iterations   = 50,
#     save_every   = 10,
#     )

#     # result = run_optimization_nst(
#     #     content_path = "../../outputs/test_imgs/content.jpg",
#     #     style_path   = "../../outputs/test_imgs/style.jpg",
#     #     output_path  = "../../outputs/test_imgs/opt_result.jpg",
#     #     img_size     = 128,     # small for quick test
#     #     iterations   = 50,      # only 50 for smoke test
#     #     save_every   = 10,
#     # )
#     print("\n  Result metrics:", result)
