"""Action space layout for /data/pan-2 episodes (act dim 25).

Derived by correlating act.npy columns against raw/*.jsonl key events over
10 episodes with frame alignment (2026-07-15). Do not re-guess: cols 0-22 are
binary, cols 23-24 are camera. Frame-aligned recall/precision vs jsonl:

  col 0: escape         rec=0.67  (GUI ticks can desync from img frames)
  col 1: s (back)       rec=0.87
  col 2: DEAD
  col 3: w (forward)    rec=0.94
  col 4-12: DEAD        (presumed hotbar 1-9; never pressed in this data)
  col 13: e (inventory) rec=0.72
  col 14: space (jump)  rec=0.87
  col 15: a (left)      rec=0.81
  col 16: d (right)     rec=0.79
  col 17: lshift (sneak)  rec=0.97
  col 18: lctrl (sprint)  rec=0.96
  col 19: DEAD
  col 20: mouse.0 (attack) rec=0.98
  col 21: mouse.1 (use)    rec=0.69
  col 22: DEAD
  col 23: camera dx, quantized to 0.1 steps in [-1, 1] (21 bins)
  col 24: camera dy, quantized to 0.1 steps in [-1, 1] (21 bins)

12 of 23 button columns are dead in the current dataset. Dimension stays 23
(data-shaped); model output for dead columns learns the constant-0 target.
"""

from __future__ import annotations

DISCRETE_NAMES = (
    "escape",      # 0
    "back",        # 1
    "dead_2",      # 2
    "forward",     # 3
    "hotbar_1",    # 4  (dead)
    "hotbar_2",    # 5  (dead)
    "hotbar_3",    # 6  (dead)
    "hotbar_4",    # 7  (dead)
    "hotbar_5",    # 8  (dead)
    "hotbar_6",    # 9  (dead)
    "hotbar_7",    # 10 (dead)
    "hotbar_8",    # 11 (dead)
    "hotbar_9",    # 12 (dead)
    "inventory",   # 13
    "jump",        # 14
    "left",        # 15
    "right",       # 16
    "sneak",       # 17
    "sprint",      # 18
    "dead_19",     # 19
    "attack",      # 20
    "use",         # 21
    "dead_22",     # 22
)

N_DISCRETE = len(DISCRETE_NAMES)  # 23, matches act.npy cols 0-22

# Indices that ever fire in the current dataset (10 live buttons).
ACTIVE_DISCRETE = (0, 1, 3, 13, 14, 15, 16, 17, 18, 20, 21)

# Camera columns are the LAST TWO of the 25-dim vector, measured.
MOUSE_DIM = 2
CAMERA_STEP = 0.1  # values are multiples of 0.1 in [-1, 1]
