"""Latent VLA: every input becomes a few latent vectors; a small transformer reads them
together with learned action queries and outputs a chunk of continuous actions.

    RGB   -> CNN ─┐
    Depth -> CNN ─┴─ add per patch -> Perceiver pooling -> n_vis visual latents
    Text  -> word embeddings + transformer -> Perceiver pooling -> n_lang latents
    Proprio -> MLP -> 1 latent
    [lang | vis | proprio | action queries] -> transformer core -> action chunk
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .env import WORKSPACE
from .text import MAX_LEN, PAD, VOCAB


@dataclass
class Config:
    obs: str = "rgbd"        # "rgbd", "rgb" or "state"
    d: int = 256
    heads: int = 8
    layers: int = 4
    n_vis: int = 16
    n_lang: int = 4
    horizon: int = 16
    action_dim: int = 2
    dropout: float = 0.1


def conv_encoder(in_ch, d):
    def block(i, o):
        return [nn.Conv2d(i, o, 3, stride=2, padding=1), nn.GroupNorm(8, o), nn.GELU()]
    # 84 -> 42 -> 21 -> 11: an 11x11 grid of patch features
    return nn.Sequential(*block(in_ch, 32), *block(32, 64), *block(64, 128), nn.Conv2d(128, d, 1))


class CrossPool(nn.Module):
    """Perceiver-style pooling: learned queries attend to a set of inputs."""

    def __init__(self, n_queries, d, heads):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.ln_q, self.ln_x, self.ln_m = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pad_mask=None):
        q = self.queries.expand(x.shape[0], -1, -1)
        kv = self.ln_x(x)
        q = q + self.attn(self.ln_q(q), kv, kv, key_padding_mask=pad_mask, need_weights=False)[0]
        return q + self.mlp(self.ln_m(q))


class LatentVLA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d

        # Vision (or true-state) latents
        if cfg.obs == "state":
            self.obj_in = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
            self.obj_id = nn.Parameter(torch.randn(6, d) * 0.02)   # pusher, 3 blocks, 2 pads
            n_obs = 6
        else:
            self.rgb_enc = conv_encoder(3, d)
            self.depth_enc = conv_encoder(1, d) if cfg.obs == "rgbd" else None
            self.patch_pos = nn.Parameter(torch.randn(11 * 11, d) * 0.02)
            self.vis_pool = CrossPool(cfg.n_vis, d, cfg.heads)
            n_obs = cfg.n_vis

        # Language latents
        self.word_emb = nn.Embedding(len(VOCAB), d, padding_idx=PAD)
        self.word_pos = nn.Parameter(torch.randn(MAX_LEN, d) * 0.02)
        self.text_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout, "gelu",
                                       batch_first=True, norm_first=True), 2,
            enable_nested_tensor=False)
        self.lang_pool = CrossPool(cfg.n_lang, d, cfg.heads)

        # Proprioception latent
        self.proprio_in = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))

        # Core
        self.action_queries = nn.Parameter(torch.randn(cfg.horizon, d) * 0.02)
        n_tokens = cfg.n_lang + n_obs + 1 + cfg.horizon
        self.slot_pos = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout, "gelu",
                                       batch_first=True, norm_first=True), cfg.layers,
            enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.action_dim))

    def encode_obs(self, b):
        if self.cfg.obs == "state":
            objs = b["state"].view(-1, 6, 2) / WORKSPACE
            return self.obj_in(objs) + self.obj_id
        feat = self.rgb_enc(b["rgb"])
        if self.depth_enc is not None:
            feat = feat + self.depth_enc(b["depth"])
        patches = feat.flatten(2).transpose(1, 2) + self.patch_pos   # (B, 121, d)
        return self.vis_pool(patches)

    def encode_lang(self, tokens):
        pad = tokens == PAD
        x = self.word_emb(tokens) + self.word_pos
        x = self.text_enc(x, src_key_padding_mask=pad)
        return self.lang_pool(x, pad_mask=pad)

    def forward(self, b):
        lang = self.encode_lang(b["tokens"])
        obs = self.encode_obs(b)
        prop = self.proprio_in(b["proprio"] / WORKSPACE)[:, None]
        q = self.action_queries.expand(lang.shape[0], -1, -1)
        x = torch.cat([lang, obs, prop, q], 1) + self.slot_pos
        x = self.core(x)
        return self.out(x[:, -self.cfg.horizon:])


def save_checkpoint(path, model, extra=None):
    torch.save({"cfg": asdict(model.cfg), "state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = LatentVLA(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
