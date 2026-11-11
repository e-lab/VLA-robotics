"""Scripted pushing expert that uses the true simulator state.

Stateless: every step it looks at where the pusher is relative to the block and the
push direction (block -> pad), then either walks around the block or pushes.
"""

import numpy as np

from .env import BLOCK_HALF, MAX_STEP, PUSHER_RADIUS

CONTACT = BLOCK_HALF * 1.45 + PUSHER_RADIUS   # pusher-center distance when touching the block
CLEAR = CONTACT + 0.03                         # lateral distance for walking around the block


def expert_action(pusher, block, pad):
    to_pad = pad - block
    dist = np.linalg.norm(to_pad)
    if dist < 0.012:
        return np.zeros(2)
    d = to_pad / dist                  # push direction
    n = np.array([-d[1], d[0]])        # perpendicular
    rel = pusher - block
    s, l = rel @ d, rel @ n            # along-push and lateral coordinates
    side = 1.0 if l >= 0 else -1.0

    if s < -CONTACT + 0.012 and abs(l) < 0.012:
        # Behind the block and aligned: push, steering to stay on the push line.
        speed = min(MAX_STEP * 0.7, 0.4 * dist + 0.004)
        cmd = d * speed - n * l * 0.8
    elif s < -CONTACT + 0.005:
        # Behind the block but off the line: move onto the line behind it.
        cmd = (block - d * (CONTACT + 0.012)) - pusher
    elif abs(l) < CLEAR:
        # Beside or in front of the block and too close: step sideways away from it.
        cmd = (block + n * side * (CLEAR + 0.01) + d * s) - pusher
    else:
        # Far enough to the side: walk back past the block.
        cmd = (block + n * side * CLEAR - d * (CONTACT + 0.02)) - pusher

    norm = np.linalg.norm(cmd)
    if norm > MAX_STEP:
        cmd = cmd / norm * MAX_STEP
    return cmd / MAX_STEP
