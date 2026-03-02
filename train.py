"""
train.py — Training loop for the 2048 DQN agent.

Usage
─────
  python train.py                        # fresh run, 50 000 episodes
  python train.py -r models/ckpt.pt      # resume from checkpoint
  python train.py -e 100000 -s 2000      # 100k episodes, save every 2k
  python train.py --help

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REWARD SHAPING  ←  edit the block below
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from agent import RewardConfig
MY_REWARD = RewardConfig(
    merge_weight    = 1.0,
    survival_bonus  = 1.0,
    log_scale       = True,
    empty_weight    = 0.1,   # +0.1 per free cell
    monotone_weight = 0.0,   # set >0 to reward sorted boards
)
Then pass it: train(reward_cfg=MY_REWARD)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFORMANCE BOTTLENECKS (ranked)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. learn() called every single step  →  pass learn_every=2 or 4 to DQNAgent
2. encode() called twice per step    →  fixed: reuse next_state as next state
3. Single-sample inference in act()  →  inherent to online RL
4. MPS/CPU transfers per step        →  already batched in ReplayBuffer.sample()
5. valid_actions() full board scan   →  already fast numpy; not a bottleneck
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import time
from pathlib import Path

import numpy as np

from agent import DQNAgent, TwoZeroFourEightNet, RewardConfig, DEFAULT_REWARD_CFG, DEVICE


# ─────────────────────────────── ANSI helpers ─────────────────────────────────
_ESC = "\033["

def _c(code: str, text: str) -> str: return f"{_ESC}{code}m{text}{_ESC}0m"

def dim(t: str) -> str:      return _c("2",    t)
def green(t: str) -> str:    return _c("32",   t)
def yellow(t: str) -> str:   return _c("33",   t)
def magenta(t: str) -> str:  return _c("35",   t)
def cyan(t: str) -> str:     return _c("36",   t)
def white(t: str) -> str:    return _c("37",   t)
def bred(t: str) -> str:     return _c("1;31", t)
def bgreen(t: str) -> str:   return _c("1;32", t)
def byellow(t: str) -> str:  return _c("1;33", t)
def bblue(t: str) -> str:    return _c("1;34", t)
def bmagenta(t: str) -> str: return _c("1;35", t)
def bcyan(t: str) -> str:    return _c("1;36", t)
def bwhite(t: str) -> str:   return _c("1;37", t)

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

def vis(s: str) -> int:
    """Visible (non-ANSI) length of a string."""
    return len(_ANSI_RE.sub('', s))


# ─────────────────────────────── layout constants ─────────────────────────────
#
#  All widths are VISIBLE character counts (ANSI codes don't count).
#  Rule: always rjust/center/ljust the PLAIN string, then wrap in color.
#  For dynamic content with mixed colors, use vis() to measure then pad.
#
W_EP   = 15   # episode range  "  1,000 – 1,025 "
W_ASC  =  9   # avg score
W_MSC  =  9   # max score
W_MV   =  7   # avg moves
W_TILE = 11   # med + max tiles "  128   512"
W_EPS  =  7   # epsilon
W_BUF  =  7   # buffer size
W_LOSS =  8   # loss

_SEP = dim(" │ ")   # 3 visible chars

# Total visible table width
TABLE_W = W_EP + W_ASC + W_MSC + W_MV + W_TILE + W_EPS + W_BUF + W_LOSS + 7 * 3  # = 94

# Box inner content width  (TABLE_W − 4 for "  ║" left and "║" right margins)
BOX_IN = TABLE_W - 4  # = 90


# ─────────────────────────────── box primitives ───────────────────────────────
def _box_row(content: str) -> str:
    """Pad colored content to exactly BOX_IN visible chars, then wrap in box walls."""
    pad = BOX_IN - vis(content)
    return bwhite("  ║") + content + " " * max(0, pad) + bwhite("║")

def _box_blank() -> str:
    return bwhite("  ║") + " " * BOX_IN + bwhite("║")


# ─────────────────────────────── tile / score coloring ────────────────────────
def _tile_color(tile: int, w: int = 5) -> str:
    s = str(tile).rjust(w)
    if tile >= 2048: return bred(s)
    if tile >= 1024: return byellow(s)
    if tile >=  512: return yellow(s)
    if tile >=  256: return bgreen(s)
    if tile >=  128: return green(s)
    if tile >=   64: return cyan(s)
    return white(s)

_session_best: int = 0

def _score_color(s: float, w: int = 9) -> str:
    global _session_best
    fmt = f"{int(s):>{w},}"
    if s >= _session_best:
        _session_best = int(s)
        return byellow(fmt)
    if s > _session_best * 0.75:
        return yellow(fmt)
    return white(fmt)


# ─────────────────────────────── table ───────────────────────────────────────
def _hcol(label: str, w: int, fn: "function") -> str:  # type: ignore[type-arg]
    """Center plain label to width w, THEN colorize — avoids ANSI bloat in .center()."""
    return fn(label.center(w))

HEADER = (
    _hcol("EPISODE",   W_EP,   bcyan)
    + _SEP + _hcol("AVG SCORE", W_ASC,  byellow)
    + _SEP + _hcol("MAX SCORE", W_MSC,  bwhite)
    + _SEP + _hcol("MOVES",     W_MV,   bblue)
    + _SEP + _hcol("MED  MAX",  W_TILE, byellow)
    + _SEP + _hcol("ε",         W_EPS,  bmagenta)
    + _SEP + _hcol("BUF",       W_BUF,  dim)
    + _SEP + _hcol("LOSS",      W_LOSS, dim)
)
DIVIDER = dim("─" * TABLE_W)


def _fmt_row(
    ep_lo:     int,
    ep_hi:     int,
    scores:    list[int],
    moves:     list[int],
    tiles:     list[int],
    eps:       float,
    buf:       int,
    loss:      float,
    elapsed_s: float = 0.0,
) -> str:
    ep  = cyan(f"{ep_lo:>6,}") + dim(" – ") + cyan(f"{ep_hi:<6,}")
    asc = _score_color(float(np.mean(scores)), W_ASC)
    msc = bwhite(f"{max(scores):>{W_MSC},}")
    mv  = bblue(f"{float(np.mean(moves)):>{W_MV}.1f}")
    med = _tile_color(int(np.median(tiles)), 5)
    mx  = _tile_color(max(tiles), 5)
    e   = magenta(f"{eps:>{W_EPS}.4f}")
    b   = dim(f"{buf:>{W_BUF},}")
    lo  = dim(f"{loss:>{W_LOSS}.4f}") if loss else dim(f"{'—':>{W_LOSS}}")
    return (
        ep
        + _SEP + asc
        + _SEP + msc
        + _SEP + mv
        + _SEP + med + " " + mx
        + _SEP + e
        + _SEP + b
        + _SEP + lo
    )


# ─────────────────────────────── startup banner ───────────────────────────────
def _startup_banner(
    device:      str,
    n_params:    int,
    n_episodes:  int,
    learn_every: int,
    models_dir:  str,
    cfg:         RewardConfig,
) -> str:
    def kv(key: str, val: str, note: str = "") -> str:
        content = f"  {bwhite(key.ljust(20))}{bgreen(val)}"
        if note:
            content += dim(f"  {note}")
        return _box_row(content)

    learn_note = (
        "← every step  (most stable, slowest)"
        if learn_every == 1
        else f"← every {learn_every} steps  (faster)"
    )

    lines = [
        "",
        bwhite("  ╔" + "═" * BOX_IN + "╗"),
        _box_row(bcyan("  2048  ·  Deep Q-Network  Trainer".center(BOX_IN - 2))),
        _box_blank(),
        kv("device",       device),
        kv("parameters",   f"{n_params:,}"),
        kv("episodes",     f"{n_episodes:,}"),
        kv("learn_every",  str(learn_every),  learn_note),
        kv("models dir",   models_dir),
        _box_blank(),
        _box_row(dim("  Reward config".center(BOX_IN - 2))),
        _box_blank(),
        kv("merge_weight",    str(cfg.merge_weight)),
        kv("survival_bonus",  str(cfg.survival_bonus)),
        kv("log_scale",       str(cfg.log_scale)),
        kv("empty_weight",    str(cfg.empty_weight)),
        kv("monotone_weight", str(cfg.monotone_weight)),
        _box_blank(),
        bwhite("  ╚" + "═" * BOX_IN + "╝"),
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────── eval banner ──────────────────────────────────
def _eval_banner(ep: int, stats: dict[str, object], elapsed_s: float, cfg: RewardConfig) -> str:
    def stat_line(*pairs: tuple[str, str]) -> str:
        content = "  " + "   ".join(f"{bgreen(k)}: {v}" for k, v in pairs)
        return _box_row(content)

    tc: dict[int, int] = stats["tile_counts"]  # type: ignore[assignment]
    tile_bar = "  ".join(
        f"{_tile_color(t, 4)}{dim('×')}{dim(str(n))}"
        for t, n in sorted(tc.items(), reverse=True)[:6]
    )
    cfg_str = (
        f"merge×{cfg.merge_weight}  "
        f"survival={cfg.survival_bonus}  "
        f"empty×{cfg.empty_weight}  "
        f"mono×{cfg.monotone_weight}  "
        f"log={cfg.log_scale}"
    )

    mean_score: float = stats["mean_score"]  # type: ignore[assignment]
    max_score:  int   = stats["max_score"]   # type: ignore[assignment]
    n_games:    int   = stats["n"]           # type: ignore[assignment]
    mean_tile:  float = stats["mean_tile"]   # type: ignore[assignment]
    max_tile:   int   = stats["max_tile"]    # type: ignore[assignment]

    lines = [
        "",
        bwhite("  ╔" + "═" * BOX_IN + "╗"),
        _box_row(bcyan(f"  EVALUATION  @  episode {ep:,}".center(BOX_IN - 2))),
        _box_blank(),
        stat_line(
            ("avg score", byellow(f"{mean_score:>10,.0f}")),
            ("max score", byellow(f"{max_score:>10,}")),
            ("games",     white(str(n_games))),
            ("time",      dim(f"{elapsed_s:.1f}s")),
        ),
        stat_line(
            ("avg tile", byellow(f"{mean_tile:>8,.1f}")),
            ("max tile", _tile_color(max_tile, 5)),
        ),
        _box_blank(),
        _box_row(f"  {bgreen('tile dist')}:  {tile_bar}"),
        _box_blank(),
        _box_row(dim(f"  {cfg_str}")),
        bwhite("  ╚" + "═" * BOX_IN + "╝"),
        "",
    ]
    return "\n".join(lines)


def _save_banner(path: str, ep: int) -> str:
    return (
        "  " + bgreen("✔") + dim("  checkpoint saved → ")
        + cyan(path) + dim(f"  (ep {ep:,})")
    )


# ─────────────────────────────── training loop ────────────────────────────────
def train(
    n_episodes:  int              = 50_000,
    window:      int              = 25,
    eval_every:  int              = 500,
    eval_n:      int              = 50,
    save_every:  int              = 1_000,
    resume:      str | None       = None,
    models_dir:  str              = "models",
    reward_cfg:  RewardConfig | None = None,
    learn_every: int              = 1,
) -> DQNAgent:
    Path(models_dir).mkdir(exist_ok=True)
    cfg = reward_cfg or DEFAULT_REWARD_CFG

    net_params = sum(p.numel() for p in TwoZeroFourEightNet().parameters())
    print(_startup_banner(str(DEVICE), net_params, n_episodes, learn_every, models_dir + "/", cfg))

    agent    = DQNAgent(reward_cfg=cfg, learn_every=learn_every)
    start_ep = 1

    if resume:
        agent.load(resume)
        start_ep = agent.episode_count + 1
        print(f"  {bgreen('resumed')} from {cyan(resume)}")
        print(f"  starting at episode {cyan(str(start_ep))}   ε = {magenta(f'{agent.eps:.4f}')}")
        print()

    w_scores: list[int]   = []
    w_moves:  list[int]   = []
    w_tiles:  list[int]   = []
    w_losses: list[float] = []
    w_start_ep            = start_ep

    best_eval_score: float = 0.0
    header_every           = 20
    row_count              = 0
    t_train_start          = time.perf_counter()
    t_window_start         = time.perf_counter()

    interrupted = False
    def _sigint(sig: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        print()
        print(f"  {yellow('⚠')}  interrupt received — saving and exiting…")
    signal.signal(signal.SIGINT, _sigint)

    print(HEADER)
    print(DIVIDER)

    for ep in range(start_ep, start_ep + n_episodes):
        if interrupted:
            break

        score, max_tile, steps = agent.run_episode(train=True)
        w_scores.append(score)
        w_moves.append(steps)
        w_tiles.append(max_tile)
        if agent.last_loss:
            w_losses.append(agent.last_loss)

        if len(w_scores) >= window:
            if row_count > 0 and row_count % header_every == 0:
                print()
                print(HEADER)
                print(DIVIDER)

            elapsed_window = time.perf_counter() - t_window_start
            print(_fmt_row(
                ep_lo     = w_start_ep,
                ep_hi     = ep,
                scores    = w_scores,
                moves     = w_moves,
                tiles     = w_tiles,
                eps       = agent.eps,
                buf       = len(agent.buffer),
                loss      = float(np.mean(w_losses)) if w_losses else 0.0,
                elapsed_s = elapsed_window,
            ))
            row_count   += 1
            w_start_ep   = ep + 1
            w_scores.clear(); w_moves.clear(); w_tiles.clear(); w_losses.clear()
            t_window_start = time.perf_counter()

        if ep % eval_every == 0:
            t0      = time.perf_counter()
            stats   = agent.evaluate(eval_n)
            elapsed = time.perf_counter() - t0
            print(_eval_banner(ep, stats, elapsed, cfg))
            print(HEADER)
            print(DIVIDER)
            row_count = 0

            if stats["mean_score"] > best_eval_score:
                best_eval_score = float(stats["mean_score"])
                best_path = os.path.join(models_dir, "best.pt")
                agent.save(best_path)
                print(
                    f"  {bred('★')}  new best eval score "
                    f"{byellow(f'{best_eval_score:,.0f}')} → {cyan(best_path)}"
                )

        if ep % save_every == 0:
            ckpt_path = os.path.join(models_dir, f"ckpt_ep{ep:06d}.pt")
            agent.save(ckpt_path)
            print(_save_banner(ckpt_path, ep))

    final_path = os.path.join(models_dir, "final.pt")
    agent.save(final_path)
    total = time.perf_counter() - t_train_start
    h, m, s = int(total // 3600), int((total % 3600) // 60), int(total % 60)
    print()
    print(DIVIDER)
    print(
        f"  {bgreen('done')}  │  "
        f"time: {cyan(f'{h:02d}:{m:02d}:{s:02d}')}  │  "
        f"model: {cyan(final_path)}"
    )
    print()

    return agent


# ─────────────────────────────── CLI ──────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the 2048 DQN agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("-e", "--episodes",    type=int,   default=50_000, metavar="N",
                   help="total training episodes (default: 50000)")
    p.add_argument("-r", "--resume",      type=str,   default=None,   metavar="PATH",
                   help="resume from checkpoint .pt file")
    p.add_argument("-w", "--window",      type=int,   default=25,     metavar="N",
                   help="print stats every N games (default: 25)")
    p.add_argument("--eval-every",        type=int,   default=500,    metavar="N",
                   help="run greedy evaluation every N episodes")
    p.add_argument("--eval-n",            type=int,   default=50,     metavar="N",
                   help="number of greedy eval games per evaluation")
    p.add_argument("--save-every",        type=int,   default=1_000,  metavar="N",
                   help="save checkpoint every N episodes")
    p.add_argument("--models-dir",        type=str,   default="models", metavar="DIR",
                   help="directory for checkpoints (default: models/)")
    p.add_argument("--learn-every",       type=int,   default=1,      metavar="N",
                   help="gradient update every N env steps (default: 1)\n"
                        "set to 2–4 for a significant training speedup")
    p.add_argument("--survival-bonus",    type=float, default=1.0,    metavar="F",
                   help="flat per-step survival reward (default: 1.0)")
    p.add_argument("--empty-weight",      type=float, default=0.1,    metavar="F",
                   help="reward per free cell after each move (default: 0.1)")
    p.add_argument("--monotone-weight",   type=float, default=0.0,    metavar="F",
                   help="reward for monotone board ordering (default: 0.0)")
    p.add_argument("--no-log-scale",      action="store_true",
                   help="disable log1p reward scaling (not recommended)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(
        n_episodes  = args.episodes,
        window      = args.window,
        eval_every  = args.eval_every,
        eval_n      = args.eval_n,
        save_every  = args.save_every,
        resume      = args.resume,
        models_dir  = args.models_dir,
        learn_every = args.learn_every,
        reward_cfg  = RewardConfig(
            survival_bonus  = args.survival_bonus,
            empty_weight    = args.empty_weight,
            monotone_weight = args.monotone_weight,
            log_scale       = not args.no_log_scale,
        ),
    )