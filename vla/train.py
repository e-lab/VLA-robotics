"""Behavior cloning of action chunks (L1 loss), then closed-loop evaluation."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .dataset import DemoData
from .evaluate import evaluate, print_summary
from .model import Config, LatentVLA, save_checkpoint


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/push_v0")
    p.add_argument("--obs", default="rgbd", choices=["rgbd", "rgb", "state"])
    p.add_argument("--out", default="runs/rgbd")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--eval-episodes", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device()
    os.makedirs(args.out, exist_ok=True)

    data = DemoData(args.data, args.horizon)
    print(data, "| device:", device)
    model = LatentVLA(Config(obs=args.obs, horizon=args.horizon)).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0))))

    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.choice(data.train_idx, args.batch)
        b = data.batch(idx, device, augment=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type != "cpu"):
            pred = model(b)
        loss = (pred.float() - b["actions"]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vidx = rng.choice(data.val_idx, 512)
                vb = data.batch(vidx, device)
                val = (model(vb) - vb["actions"]).abs().mean().item()
            model.train()
            rate = step / (time.time() - t0)
            print(f"step {step:6d}  train L1 {loss.item():.4f}  val L1 {val:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.1e}  {rate:.1f} it/s", flush=True)
            log.append({"step": step, "train": loss.item(), "val": val})

    save_checkpoint(os.path.join(args.out, "model.pt"), model, {"args": vars(args)})
    print(f"saved {args.out}/model.pt; evaluating in MuJoCo...")
    summary = evaluate(model, device, args.eval_episodes)
    print_summary(summary)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"args": vars(args), "log": log, "eval": summary}, f, indent=2)


if __name__ == "__main__":
    main()
