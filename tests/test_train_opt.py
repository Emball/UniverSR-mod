import logging, os, sys, tempfile
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))
from train_opt import ComboOptimizer, freeze_by_prefix, make_optimizer, make_scheduler, resolve_precision

logging.basicConfig(level=logging.WARNING)


class Net(pl.LightningModule):
    def __init__(self, combo):
        super().__init__()
        self.a, self.b = torch.nn.Linear(4, 4), torch.nn.Linear(4, 1)
        self.combo = combo
    def training_step(self, batch, i):
        x, y = batch
        return torch.nn.functional.mse_loss(self.b(torch.relu(self.a(x))), y)
    def configure_optimizers(self):
        o = ComboOptimizer([torch.optim.AdamW(self.a.parameters(), lr=1e-2), torch.optim.AdamW(self.b.parameters(), lr=1e-2)]) if self.combo \
            else make_optimizer(self.parameters(), {"type": "adamw", "lr": 1e-2})
        s = make_scheduler(o, {"type": "cosine", "warmup_steps": 2, "min_lr_ratio": 0.1}, 20)
        return [o], [{"scheduler": s, "interval": "step"}]


def fit(combo, accum):
    torch.manual_seed(0)
    ds = TensorDataset(torch.randn(64, 4), torch.randn(64, 1))
    m = Net(combo)
    with tempfile.TemporaryDirectory() as d:
        t = pl.Trainer(max_steps=10, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, enable_model_summary=False, accumulate_grad_batches=accum,
                       gradient_clip_val=1.0, default_root_dir=d)
        t.fit(m, DataLoader(ds, batch_size=4))
        ckpt = os.path.join(d, "x.ckpt"); t.save_checkpoint(ckpt)
        sd = torch.load(ckpt, weights_only=False)
        assert sd["optimizer_states"] and sd["lr_schedulers"]
        t2 = pl.Trainer(max_steps=12, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False, accumulate_grad_batches=accum, default_root_dir=d)
        t2.fit(Net(combo), DataLoader(ds, batch_size=4), ckpt_path=ckpt)
        return t.global_step, t2.global_step, m.trainer.optimizers[0].param_groups[0]["lr"]


for combo in (False, True):
    for accum in (1, 2):
        g1, g2, lr = fit(combo, accum)
        assert g1 == 10 and g2 == 12, (g1, g2)
        print(f"  combo={combo} accum={accum}: fit+save+resume ok, steps {g1}->{g2}, lr {lr:.5f}")

o = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
s = make_scheduler(o, {"type": "cosine", "warmup_steps": 4, "min_lr_ratio": 0.0}, 100)
lrs = []
for _ in range(100):
    lrs.append(o.param_groups[0]["lr"]); o.step(); s.step()
assert abs(lrs[0] - 0.25) < 1e-6 and abs(lrs[3] - 1.0) < 1e-6 and lrs[-1] < 1e-3 and max(lrs) <= 1.0
print("  cosine warmup/decay shape ok:", [round(lrs[i], 3) for i in (0, 3, 50, 99)])
try:
    make_scheduler(o, {"type": "cosine", "warmup_steps": 50}, 20); raise SystemExit("should have raised")
except ValueError:
    print("  cosine rejects total<=warmup")

net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
fz, tot = freeze_by_prefix(net, ["0."])
assert fz == 6 and not net[0].weight.requires_grad and net[1].weight.requires_grad
try:
    freeze_by_prefix(net, ["nope"]); raise SystemExit("should have raised")
except ValueError:
    print("  freeze ok, typo prefix rejected")

assert resolve_precision("16-mixed") == "16-mixed" and resolve_precision("auto") == "32-true"
print("  precision: explicit honoured, auto -> 32-true on CPU")
print("all passed")
