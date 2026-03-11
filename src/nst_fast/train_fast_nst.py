"""
Fast NST Training Script — StyleSense (Fixed v2)
Owner: Shubhansh Gupta
"""

import os, sys, time, urllib.request
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "../extractor"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../nst_optimization"))

from load_checkpoint import load_extractor
from losses          import NSTLoss
from generator       import FastNSTGenerator

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def normalize(x, device):
    return (x - MEAN.to(device)) / STD.to(device)

def load_style_image(path, size, device):
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)

def download_style_image(save_path):
    """Download Van Gogh with browser-like headers."""
    import urllib.request
    url = ("https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/"
           "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/"
           "800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        print("  Downloading Van Gogh style image...")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        })
        with urllib.request.urlopen(req) as resp, \
             open(save_path, "wb") as f:
            f.write(resp.read())
        print(f"  Downloaded: {save_path} ✅")
    else:
        print(f"  Style image found: {save_path} ✅")



def train_fast_nst(
    style_path     = "../../outputs/test_imgs/vangogh_style.jpg",
    checkpoint     = "../../checkpoints/best_extractor.pth",
    save_dir       = "../../checkpoints",
    data_dir       = "../../data",
    img_size       = 64,
    batch_size     = 16,
    epochs         = 2,
    lr             = 1e-3,
    content_weight = 1.0,
    style_weight   = 50.0,    # tuned for gram/C normalization
    tv_weight      = 1e-4,
    device         = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device      : {device}")
    print(f"  Style weight: {style_weight}")
    print(f"  Img size    : {img_size}x{img_size}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("../../outputs/test_imgs", exist_ok=True)

    # Download real style image if not present
    download_style_image(style_path)

    # Load extractor
    extractor, _ = load_extractor(checkpoint, device=device)
    extractor = extractor.to(device).eval()
    for p in extractor.parameters():
        p.requires_grad = False

    # Style features
    style_img = load_style_image(style_path, img_size, device)
    with torch.no_grad():
        style_features = extractor.get_feature_maps(style_img)

    # Debug: print gram values to verify non-zero
    from losses import gram_matrix
    with torch.no_grad():
        g_test = gram_matrix(style_features["block1"])
        print(f"\n  [Debug] Style gram (block1) mean : {g_test.abs().mean().item():.4f}")
        g_test2 = gram_matrix(style_features["block3"])
        print(f"  [Debug] Style gram (block3) mean : {g_test2.abs().mean().item():.4f}")
        if g_test.abs().mean().item() == 0:
            print("  [WARN] Gram matrix is zero! Check style image & gram_matrix()")
        else:
            print("  [OK]   Gram matrices are non-zero ✅")

    # Dataset
    print("\n  Loading CIFAR-10 into RAM...")
    tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    raw = torchvision.datasets.CIFAR10(data_dir, train=True,
                                        download=True, transform=tf)
    all_imgs = torch.stack([raw[i][0] for i in range(min(10000, len(raw)))])
    loader = DataLoader(TensorDataset(all_imgs),
                        batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"  Dataset: {len(all_imgs)} images ✅")

    # Generator
    generator = FastNSTGenerator(n_residual=5).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    criterion = NSTLoss(content_weight, style_weight, tv_weight)

    print(f"\n  {'Ep':<4}{'Batch':<8}{'Total':>10}"
          f"{'Content':>12}{'Style':>12}{'TV':>12}{'Time':>8}")
    print(f"  {'-'*66}")

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        generator.train()
        for bi, (cb,) in enumerate(loader):
            cb = cb.to(device)
            cb_norm = normalize(cb, device)

            gen_out  = generator(cb_norm)
            gen_norm = normalize((gen_out + 1.0) / 2.0, device)

            gen_feat  = extractor.get_feature_maps(gen_norm)
            cont_feat = extractor.get_feature_maps(cb_norm)

            style_batch = {
                k: v.expand(cb.size(0), -1, -1, -1).detach()
                for k, v in style_features.items()
            }

            total, lc, ls, lt = criterion(
                gen_out, gen_feat, cont_feat, style_batch)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            if bi % 50 == 0:
                el = time.time() - t_start
                print(f"  {epoch:<4}{bi:<8}{total.item():>10.4f}"
                      f"{lc.item():>12.6f}{ls.item():>12.6f}"
                      f"{lt.item():>12.6f}{el:>7.1f}s")

        ckpt = os.path.join(save_dir, f"fast_nst_epoch{epoch}.pth")
        torch.save({"epoch": epoch,
                    "model_state_dict": generator.state_dict(),
                    "style_path": style_path}, ckpt)
        print(f"\n  Epoch {epoch} done | Checkpoint: {ckpt}\n")

    print(f"  Total time: {(time.time()-t_start)/60:.1f} mins")
    return generator


if __name__ == "__main__":
    train_fast_nst(
        img_size    = 64,
        batch_size  = 16,
        epochs      = 2,
        style_weight= 50.0,
    )
