# test_speed.py  (src/extractor ke andar banao)
import time, torch
from vgg_like_cnn import VGGLikeExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGLikeExtractor(num_classes=10).to(device)
x = torch.randn(64, 3, 64, 64).to(device)  # batch=64, 64x64

# warmup
for _ in range(5):
    _ = model(x)

torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    _ = model(x)
torch.cuda.synchronize()
t1 = time.time()

print("Per forward batch time (64x64):", (t1 - t0) / 20, "sec")
