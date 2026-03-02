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
2. encode() called twice per step    →  fixed below: reuse next_state
3. Single-sample inference in act()  →  inherent to online RL; see note in act()
4. MPS/CPU transfers per step        →  already batched in ReplayBuffer.sample()
5. valid_actions() full board scan   →  already fast numpy; not a bottleneck
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import numpy as np

from agent import DQNAgent, TwoZeroFourEightNet, RewardConfig, DEFAULT_REWARD_CFG, DEVICE

# ─────────────────────────────── ANSI helpers ─────────────────────────────────
_ESC = "\033["

def _c(code: str, text: str) -> str:   return f"{_ESC}{code}m{text}{_ESC}0m"

def dim(t):      return _c("2",    t)
def bold(t):     return _c("1",    t)
def red(t):      return _c("31",   t)
def green(t):    return _c("32",   t)
def yellow(t):   return _c("33",   t)
def blue(t):     return _c("34",   t)
def magenta(t):  return _c("35",   t)
def cyan(t):     return _c("36",   t)
def white(t):    return _c("37",   t)
def bred(t):     return _c("1;31", t)
def bgreen(t):   return _c("1;32", t)
def byellow(t):  return _c("1;33", t)
def bblue(t):    return _c("1;34", t)
def bmagenta(t): return _c("1;35", t)
def bcyan(t):    return _c("1;36", t)
def bwhite(t):   return _c("1;37", t)


# ─────────────────────────────── tile / score colouring ───────────────────────
def _tile_color(tile: int, width: int = 5) -> str:
    s = str(tile).rjust(width)
    if tile >= 2048: return bred(s)
    if tile >= 1024: return byellow(s)
    if tile >=  512: return yellow(s)
    if tile >=  256: return bgreen(s)
    if tile >=  128: return green(s)
    if tile >=   64: return cyan(s)
    return white(s)


_session_best_score: int = 0

def _score_color(s: float, width: int = 9) -> str:
    global _session_best_score
    fmt = f"{int(s):>{width},}"
    if s >= _session_best_score:
        _session_best_score = int(s)
        return byellow(fmt)
    if s > _session_best_score * 0.75:
        return yellow(fmt)
    return white(fmt)


# ─────────────────────────────── column layout ────────────────────────────────
#
#  FIX: the original code mixed raw and ANSI strings in f-strings, which caused
#  misaligned columns because ANSI escape codes are invisible but still count
#  toward Python's string length. Solution: compute the PLAIN widths first,
#  then wrap in colour AFTER rjust/ljust. All columns are fixed-width below.
#
#  Column widths (chars of visible text):
#    EPISODE   : 14  (e.g. " 1,000-1,025 ")
#    AVG SCORE :  9
#    MAX SCORE :  9
#    AVG MOVES :  9
#    MED TILE  :  5
#    MAX TILE  :  5
#    ε         :  7
#    BUF       :  8
#    LOSS      :  8
#    TIME      :  8
#
_SEP = dim(" │ ")

def _hdr_col(label: str, width: int, color_fn=bwhite) -> str:
    return color_fn(label.center(width))

HEADER_LINE = (
    _hdr_col("EPISODE",   14, bcyan)
    + _SEP + _hdr_col("AVG SCORE",  9, byellow)
    + _SEP + _hdr_col("MAX SCORE",  9, bwhite)
    + _SEP + _hdr_col("AVG MOVES",  9, bblue)
    + _SEP + _hdr_col("MED TILE",   5, byellow)
    + " " + _hdr_col("MAX TILE",   5, byellow)
    + _SEP + _hdr_col("ε",          7, bmagenta)
    + _SEP + _hdr_col("BUF",        8, dim)
    + _SEP + _hdr_col("LOSS",       8, dim)
    + _SEP + _hdr_col("TIME",       8, bgreen)
)

# Width of the visible text in HEADER_LINE (sum of col widths + separators)
# Used to draw the divider line at the same length.
_HEADER_VISIBLE_W = 14 + 9 + 9 + 9 + 5 + 1 + 5 + 7 + 8 + 8 + 8 + (9 * 3)  # 3-char seps
DIVIDER = dim("─" * 114)


def _fmt_row(
    ep_lo: int,
    ep_hi: int,
    scores: list[int],
    moves:  list[int],
    tiles:  list[int],
    eps:    float,
    buf:    int,
    loss:   float,
    elapsed_s: float = 0.0,
) -> str:
    # Episode range: fixed 14 visible chars
    ep_str = (
        cyan(f"{ep_lo:>6,}")
        + dim("–")
        + cyan(f"{ep_hi:<6,}")
    )   # 6 + 1 + 6 = 13; +1 padding on the right gives 14

    avg_sc  = _score_color(np.mean(scores), width=9) # type: ignore
    max_sc  = bwhite(f"{max(scores):>9,}")
    avg_mv  = bblue(f"{np.mean(moves):>9.1f}")
    med_til = _tile_color(int(np.median(tiles)), width=5)
    max_til = _tile_color(max(tiles), width=5)
    eps_str = magenta(f"{eps:>7.4f}")
    buf_str = dim(f"{buf:>8,}")
    loss_s  = dim(f"{loss:>8.4f}") if loss else dim(f"{'—':>8}")
    time_s  = bgreen(f"{elapsed_s:>8.2f}s")

    return (
        f" {ep_str} "
        + _SEP + avg_sc
        + _SEP + max_sc
        + _SEP + avg_mv
        + _SEP + med_til + " " + max_til
        + _SEP + eps_str
        + _SEP + buf_str
        + _SEP + loss_s
        + _SEP + time_s
    )


# ─────────────────────────────── eval banner ──────────────────────────────────
def _eval_banner(ep: int, stats: dict, elapsed_s: float, cfg: RewardConfig) -> str: # type: ignore
    tile_bar = "  ".join(
        f"{_tile_color(t, width=4)}{dim('×')}{dim(str(n))}"
        for t, n in sorted(stats["tile_counts"].items(), reverse=True)[:8]
    )
    lines = [ # type: ignore
        "",
        bwhite("  ╔" + "═" * 74 + "╗"),
        bwhite("  ║") + bcyan(f"  EVALUATION @ episode {ep:,}".center(74)) + bwhite("║"),
        bwhite("  ║") + " " * 74 + bwhite("║"),
        (bwhite("  ║") # type: ignore
            + f"  {bgreen('avg score')}: {byellow(f'{stats['mean_score']:>9,.0f}')}" # type: ignore
            + f"   {bgreen('max score')}: {byellow(f'{stats['max_score']:>9,}')}"
            + f"   {bgreen('games')}: {white(str(stats['n'])):<6}"
            + f"   elapsed: {dim(f'{elapsed_s:.1f}s')}"
            + " " * 4 + bwhite("║")),
        (bwhite("  ║")
            + f"  {bgreen('avg tile')}: {byellow(f'{stats['mean_tile']:>8,.1f}')}"
            + f"   {bgreen('max tile')}: {_tile_color(stats['max_tile'], width=5)}"
            + " " * 20 + bwhite("║")),
        bwhite("  ║") + f"  {bgreen('tile dist')}: {tile_bar}" + " " * 4 + bwhite("║"),
        bwhite("  ║") + " " * 74 + bwhite("║"),
        (bwhite("  ║")
            + dim(f"  reward: merge×{cfg.merge_weight}  survival={cfg.survival_bonus}"
                  f"  empty×{cfg.empty_weight}  mono×{cfg.monotone_weight}"
                  f"  log={cfg.log_scale}").ljust(76)
            + bwhite("║")),
        bwhite("  ╚" + "═" * 74 + "╝"),
        "",
    ]
    return "\n".join(lines)


def _save_banner(path: str, ep: int) -> str:
    return (
        "  " + bgreen("✔") + dim(" checkpoint → ")
        + cyan(path) + dim(f"  (ep {ep:,})")
    )


# ─────────────────────────────── training loop ────────────────────────────────
def train(
    n_episodes:  int  = 50_000,
    window:      int  = 25,
    eval_every:  int  = 500,
    eval_n:      int  = 50,
    save_every:  int  = 1_000,
    resume:      str | None = None,
    models_dir:  str  = "models",
    # ── reward shaping ────────────────────────────────────────────────────────
    reward_cfg:  RewardConfig | None = None,
    # ── performance ───────────────────────────────────────────────────────────
    learn_every: int  = 1,   # 1 = update every step; 2-4 = significant speedup
):
    """
    Main training loop.

    Performance tips
    ────────────────
    • learn_every=2 halves gradient overhead with negligible quality loss.
    • Larger batch_size (512→1024) improves GPU utilisation on Apple MPS.
    • Reducing eval_n from 50→20 speeds up eval checkpoints noticeably.
    • The biggest single speedup: learn_every=4 + batch_size=1024.

    Reward shaping
    ──────────────
    Pass reward_cfg=RewardConfig(...) to override any reward parameter.
    See agent.RewardConfig for all available knobs.
    """
    Path(models_dir).mkdir(exist_ok=True)
    cfg = reward_cfg or DEFAULT_REWARD_CFG

    # ── startup banner ────────────────────────────────────────────────────────
    print()
    print(bwhite("  ┌──────────────────────────────────────────────────────┐"))
    print(bwhite("  │") + bcyan("          2048  ·  Deep Q-Network  Trainer            ") + bwhite("│"))
    print(bwhite("  └──────────────────────────────────────────────────────┘"))
    print(f"  device       : {bgreen(str(DEVICE))}")
    net_params = sum(p.numel() for p in TwoZeroFourEightNet().parameters())
    print(f"  parameters   : {bgreen(f'{net_params:,}')}")
    print(f"  episodes     : {bgreen(str(n_episodes))}")
    print(f"  learn_every  : {bgreen(str(learn_every))}"
          + (dim("  ← update every step (slowest)") if learn_every == 1
             else dim(f"  ← gradient every {learn_every} steps")))
    print(f"  models dir   : {bgreen(models_dir + '/')}")
    print()
    print(f"  {bwhite('Reward config:')}")
    print(f"    merge_weight    = {byellow(str(cfg.merge_weight))}")
    print(f"    survival_bonus  = {byellow(str(cfg.survival_bonus))}")
    print(f"    log_scale       = {byellow(str(cfg.log_scale))}")
    print(f"    empty_weight    = {byellow(str(cfg.empty_weight))}")
    print(f"    monotone_weight = {byellow(str(cfg.monotone_weight))}")
    print()

    agent = DQNAgent(reward_cfg=cfg, learn_every=learn_every)
    start_ep = 1

    if resume:
        agent.load(resume)
        start_ep = agent.episode_count + 1
        print(f"  {bgreen('resumed')} from {cyan(resume)}")
        print(f"  starting at episode {cyan(str(start_ep))}  ε={magenta(f'{agent.eps:.4f}')}")
        print()

    # ── accumulators ──────────────────────────────────────────────────────────
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

    # ── graceful Ctrl-C ───────────────────────────────────────────────────────
    interrupted = False
    def _sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print()
        print(f"  {yellow('⚠')}  interrupt received — saving and exiting…")
    signal.signal(signal.SIGINT, _sigint)

    # ── header ────────────────────────────────────────────────────────────────
    print(HEADER_LINE)
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

        # ── print stats row ───────────────────────────────────────────────────
        if len(w_scores) >= window:
            if row_count > 0 and row_count % header_every == 0:
                print()
                print(HEADER_LINE)
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
            row_count    += 1
            w_start_ep    = ep + 1
            w_scores.clear(); w_moves.clear(); w_tiles.clear(); w_losses.clear()
            t_window_start = time.perf_counter()

        # ── eval ──────────────────────────────────────────────────────────────
        if ep % eval_every == 0:
            t0      = time.perf_counter()
            stats   = agent.evaluate(eval_n)
            elapsed = time.perf_counter() - t0
            print(_eval_banner(ep, stats, elapsed, cfg))
            print(HEADER_LINE)
            print(DIVIDER)
            row_count = 0

            if stats["mean_score"] > best_eval_score:
                best_eval_score = stats["mean_score"]
                best_path = os.path.join(models_dir, "best.pt")
                agent.save(best_path)
                print(f"  {bred('★')} new best eval score "
                      f"{byellow(f'{best_eval_score:,.0f}')} → {cyan(best_path)}")

        # ── checkpoint ────────────────────────────────────────────────────────
        if ep % save_every == 0:
            ckpt_path = os.path.join(models_dir, f"ckpt_ep{ep:06d}.pt")
            agent.save(ckpt_path)
            print(_save_banner(ckpt_path, ep))

    # ── final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(models_dir, "final.pt")
    agent.save(final_path)
    total = time.perf_counter() - t_train_start
    h, m, s = int(total // 3600), int((total % 3600) // 60), int(total % 60)
    print()
    print(bwhite("  " + "─" * 60))
    print(f"  {bgreen('Training complete')}  │  "
          f"total time: {cyan(f'{h:02d}:{m:02d}:{s:02d}')}  │  "
          f"final model: {cyan(final_path)}")
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
    # ── performance ────────────────────────────────────────────────────────
    p.add_argument("--learn-every",       type=int,   default=1,      metavar="N",
                   help="run gradient update every N env steps (default: 1)\n"
                        "Set to 2 or 4 for a significant training speedup.")
    # ── reward shaping ─────────────────────────────────────────────────────
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

    my_reward_cfg = RewardConfig(
        survival_bonus  = args.survival_bonus,
        empty_weight    = args.empty_weight,
        monotone_weight = args.monotone_weight,
        log_scale       = not args.no_log_scale,
    )

    train(
        n_episodes  = args.episodes,
        window      = args.window,
        eval_every  = args.eval_every,
        eval_n      = args.eval_n,
        save_every  = args.save_every,
        resume      = args.resume,
        models_dir  = args.models_dir,
        reward_cfg  = my_reward_cfg,
        learn_every = args.learn_every,
    )