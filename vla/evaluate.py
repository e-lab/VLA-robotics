"""Closed-loop evaluation in MuJoCo: success rate per (block, pad) instruction."""

import argparse
import itertools
import json
import os

import numpy as np
import torch

from .env import BLOCK_NAMES, HELDOUT_TASKS, PAD_NAMES, PushEnv
from .text import encode

MAX_STEPS = 120
EVAL_SEED_OFFSET = 1_000_000   # far away from training seeds


def obs_to_batch(obs, tokens, device):
    rgb = torch.from_numpy(obs["rgb"]).to(device).permute(2, 0, 1)[None].float() / 255.0
    depth = torch.from_numpy(obs["depth"]).to(device)[None, None]
    return {
        "rgb": rgb, "depth": depth, "tokens": tokens,
        "proprio": torch.from_numpy(obs["proprio"]).to(device)[None],
        "state": torch.from_numpy(obs["state"]).to(device)[None],
    }


@torch.no_grad()
def run_policy_episode(model, env, seed, task, device, execute, frames=None):
    obs = env.reset(seed, task=task, allow_heldout=True)
    tokens = torch.from_numpy(encode(env.instruction)).to(device)[None]
    plan, k = None, 0
    for _ in range(MAX_STEPS):
        if plan is None or k >= execute:
            plan, k = model(obs_to_batch(obs, tokens, device))[0].cpu().numpy(), 0
        if frames is not None:
            frames.append(obs["rgb"])
        obs, success = env.step(plan[k])
        k += 1
        if success:
            return True
    return False


def evaluate(model, device, episodes=120, execute=8, gif_dir=None, n_gifs=0):
    model.eval()
    env = PushEnv()
    tasks = list(itertools.product(BLOCK_NAMES, PAD_NAMES))
    results = {f"{b}->{p}": [] for b, p in tasks}
    for i in range(episodes):
        task = tasks[i % len(tasks)]
        frames = [] if gif_dir and i < n_gifs else None
        ok = run_policy_episode(model, env, EVAL_SEED_OFFSET + i, task, device, execute, frames)
        results[f"{task[0]}->{task[1]}"].append(ok)
        if frames:
            save_gif(frames, os.path.join(gif_dir, f"ep{i:02d}_{task[0]}_{task[1]}_{'ok' if ok else 'fail'}.gif"))
    seen = [x for (b, p) in tasks if (b, p) not in HELDOUT_TASKS for x in results[f"{b}->{p}"]]
    held = [x for (b, p) in tasks if (b, p) in HELDOUT_TASKS for x in results[f"{b}->{p}"]]
    summary = {
        "success_seen": float(np.mean(seen)),
        "success_heldout": float(np.mean(held)) if held else None,
        "per_task": {k: float(np.mean(v)) for k, v in results.items()},
        "episodes": episodes,
    }
    model.train()
    return summary


def save_gif(frames, path, scale=3):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imgs = [Image.fromarray(f).resize((f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)


def print_summary(s):
    print(f"success on seen tasks:     {s['success_seen']:.1%}")
    if s["success_heldout"] is not None:
        print(f"success on held-out task:  {s['success_heldout']:.1%}  {sorted(HELDOUT_TASKS)}")
    for k, v in s["per_task"].items():
        print(f"  {k:16s} {v:.1%}")


def main():
    from .model import load_checkpoint
    from .train import pick_device

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=240)
    p.add_argument("--execute", type=int, default=8, help="actions executed per predicted chunk")
    p.add_argument("--gifs", type=int, default=6, help="save GIFs of the first N episodes")
    args = p.parse_args()

    device = pick_device()
    model, _ = load_checkpoint(args.ckpt, device)
    run_dir = os.path.dirname(args.ckpt)
    s = evaluate(model, device, args.episodes, args.execute, os.path.join(run_dir, "gifs"), args.gifs)
    print_summary(s)
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(s, f, indent=2)


if __name__ == "__main__":
    main()
