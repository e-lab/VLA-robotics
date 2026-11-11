"""Word-level vocabulary for the templated commands (v0 text input)."""

import numpy as np

from .env import BLOCK_NAMES, PAD_NAMES, TEMPLATES

MAX_LEN = 10
PAD, UNK = 0, 1


def _build_vocab():
    words = set(BLOCK_NAMES) | set(PAD_NAMES)
    for t in TEMPLATES:
        words |= set(t.replace("{b}", "").replace("{p}", "").split())
    return ["<pad>", "<unk>"] + sorted(words)


VOCAB = _build_vocab()
WORD_ID = {w: i for i, w in enumerate(VOCAB)}


def encode(text):
    ids = [WORD_ID.get(w, UNK) for w in text.lower().split()][:MAX_LEN]
    return np.array(ids + [PAD] * (MAX_LEN - len(ids)), np.int64)


def encode_batch(texts):
    return np.stack([encode(t) for t in texts])
