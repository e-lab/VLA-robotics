"""In-memory demo dataset that serves (observation, instruction, action chunk) batches."""

import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .text import encode_batch


class DemoData:
    def __init__(self, path, horizon, val_frac=0.05, seed=0):
        shards = sorted(glob.glob(os.path.join(path, "shard_*.npz")))
        if not shards:
            raise FileNotFoundError(f"no shards in {path}")
        parts = {k: [] for k in ("rgb", "depth", "proprio", "state", "action", "episode")}
        meta = []
        for s in shards:
            z = np.load(s)
            for k in parts:
                v = z[k]
                parts[k].append(v + len(meta) if k == "episode" else v)
            meta += json.loads(str(z["meta"]))
        for k, v in parts.items():
            setattr(self, k, np.concatenate(v))
        self.meta = meta
        self.horizon = horizon

        # Index of the last step of each step's episode, for building padded chunks.
        n_ep = len(meta)
        ends = np.zeros(n_ep, np.int64)
        ends[self.episode] = np.arange(len(self.episode))   # last write wins = last step
        self.ep_end = ends[self.episode]
        self.tokens = encode_batch([m["instruction"] for m in meta])

        rng = np.random.default_rng(seed)
        val_eps = set(rng.choice(n_ep, max(1, int(n_ep * val_frac)), replace=False).tolist())
        is_val = np.array([e in val_eps for e in self.episode])
        self.train_idx = np.nonzero(~is_val)[0]
        self.val_idx = np.nonzero(is_val)[0]

    def __repr__(self):
        return (f"DemoData({len(self.meta)} episodes, {len(self.action)} steps, "
                f"{len(self.train_idx)} train / {len(self.val_idx)} val steps)")

    def batch(self, idx, device, augment=False):
        chunk = idx[:, None] + np.arange(self.horizon)[None]
        valid = chunk <= self.ep_end[idx][:, None]
        chunk = np.minimum(chunk, self.ep_end[idx][:, None])
        actions = self.action[chunk] * valid[..., None]   # zeros after the episode ends = stay still

        out = {
            "proprio": torch.from_numpy(self.proprio[idx]).to(device),
            "state": torch.from_numpy(self.state[idx]).to(device),
            "tokens": torch.from_numpy(self.tokens[self.episode[idx]]).to(device),
            "actions": torch.from_numpy(actions.astype(np.float32)).to(device),
        }
        rgb = torch.from_numpy(self.rgb[idx]).to(device).permute(0, 3, 1, 2).float() / 255.0
        depth = torch.from_numpy(self.depth[idx].astype(np.float32)).to(device)[:, None]
        img = torch.cat([rgb, depth], 1)
        if augment:
            img = random_shift(img)
        out["rgb"], out["depth"] = img[:, :3], img[:, 3:]
        return out


def random_shift(img, pad=4):
    """Shift each image by up to `pad` pixels (DrQ-style augmentation)."""
    b, _, h, w = img.shape
    padded = F.pad(img, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=img.device)[:h]
    ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base = torch.cat([ar, ar.transpose(1, 0)], 2).unsqueeze(0).repeat(b, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (b, 1, 1, 2), device=img.device).float() * 2.0 * eps
    return F.grid_sample(padded, base + shift, padding_mode="zeros", align_corners=False)
