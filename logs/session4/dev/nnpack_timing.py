# git 183b56c (clean), macOS arm64; output in nnpack_timing.log
"""Dev-only: median wall time of one diffusion training step (batch 100, primary config),
U-Net with NNPACK enabled vs disabled (macOS arm64, 1 torch thread)."""
import statistics, time, contextlib, json, platform
import torch
from roboforge.models.registry import build_model
from roboforge.train.config import load_config
import roboforge.models.diffusion as dm
torch.set_num_threads(1)
cfg = load_config("configs/models/diffusion_unet.yaml")
torch.manual_seed(0)
m = build_model(cfg.family, cfg.model, 19, 7, 2, 16).train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
g = torch.Generator().manual_seed(0)
oh, ov = torch.rand(100, 2, 19) * 2 - 1, torch.ones(100, 2, dtype=torch.bool)
fa, vm = torch.rand(100, 16, 7) * 2 - 1, torch.ones(100, 16, dtype=torch.bool)
orig = dm.unet_backend
def run(label, backend):
    dm.unet_backend = backend
    ts = []
    for i in range(25):
        t0 = time.perf_counter()
        loss = m.training_loss(oh, ov, fa, vm, generator=g)
        opt.zero_grad(); loss.backward(); opt.step()
        ts.append(1000 * (time.perf_counter() - t0))
    return {"label": label, "median_ms": statistics.median(ts[5:]), "n": 20}
out = [run("nnpack_disabled (unet_backend)", orig),
       run("nnpack_enabled (torch default)", contextlib.nullcontext),
       run("nnpack_disabled again", orig)]
dm.unet_backend = orig
print(json.dumps({"platform": platform.platform(), "torch": torch.__version__,
                  "nnpack_available": torch.backends.nnpack.is_available(),
                  "note": "dev-only; 1 thread; a 30k-step training run was running concurrently",
                  "results": out}, indent=1))
