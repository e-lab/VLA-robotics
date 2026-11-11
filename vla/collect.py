"""Collect scripted-expert demonstrations in parallel.

Each step stores RGB, depth, pusher position, full state, the clean expert action
and the episode's instruction. The executed action has noise added (DART-style),
so demos include small mistakes and the expert's corrections.
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from .env import BLOCK_NAMES, PAD_NAMES, PushEnv
from .expert import expert_action

MAX_STEPS = 120
EXTRA_STEPS = 3   # steps recorded after success, so the policy learns to settle


def run_episode(env, seed, noise, rng):
    obs = env.reset(seed)
    frames = {k: [] for k in ("rgb", "depth", "proprio", "state", "action")}
    after = None
    for _ in range(MAX_STEPS):
        a = expert_action(env.pusher_xy(), env.target_block_xy(), env.target_pad_xy())
        for k in ("rgb", "depth", "proprio", "state"):
            frames[k].append(obs[k])
        frames["action"].append(a.astype(np.float32))
        executed = a + rng.normal(0.0, noise, 2) if noise > 0 else a
        obs, success = env.step(executed)
        if success and after is None:
            after = EXTRA_STEPS
        if after is not None:
            after -= 1
            if after < 0:
                break
    ok = after is not None and env.is_success()
    return ok, frames


def worker(args):
    seeds, noise, out_path = args
    env = PushEnv()
    rng = np.random.default_rng(seeds[0] + 12345)
    data = {k: [] for k in ("rgb", "depth", "proprio", "state", "action", "episode")}
    meta = []
    n_fail = 0
    for seed in seeds:
        ok, frames = run_episode(env, seed, noise, rng)
        if not ok:
            n_fail += 1
            continue
        for k, v in frames.items():
            data[k].append(np.stack(v))
        data["episode"].append(np.full(len(frames["action"]), len(meta), np.int32))
        meta.append({"seed": int(seed), "instruction": env.instruction,
                     "block": BLOCK_NAMES.index(env.block), "pad": PAD_NAMES.index(env.pad)})
    arrays = {k: np.concatenate(v) for k, v in data.items()}
    arrays["depth"] = arrays["depth"].astype(np.float16)
    np.savez(out_path, **arrays, meta=json.dumps(meta))
    return len(meta), n_fail, len(arrays["action"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--out", default="data/push_v0")
    p.add_argument("--noise", type=float, default=0.25, help="std of noise added to executed actions")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seeds = np.arange(args.seed, args.seed + args.episodes)
    shards = np.array_split(seeds, args.workers * 2)
    jobs = [(s, args.noise, os.path.join(args.out, f"shard_{i:03d}.npz")) for i, s in enumerate(shards)]
    t = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(worker, jobs)
    ok = sum(r[0] for r in results)
    fail = sum(r[1] for r in results)
    steps = sum(r[2] for r in results)
    print(f"kept {ok} episodes ({fail} failed, discarded), {steps} steps, "
          f"{time.time() - t:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
