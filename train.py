"""
train.py — Training loop for the 2048 DQN agent.

Usage
─────
  python train.py                        # fresh run, 50 000 episodes
  python train.py -r models/ckpt.pt      # resume from checkpoint
  python train.py -e 100000 -s 2000      # 100k episodes, save every 2k
  python train.py --help

Output
──────
  • A stats row is printed every --window games (default 25):
        EP      AVG SCORE   MAX SCORE   AVG MOVES   MED TILE   MAX TILE   ε        BUF     LOSS
  • An evaluation banner runs every --eval-every episodes (greedy games).
  • Checkpoints are saved to models/ every --save-every episodes.
  • Best-model (highest eval avg score) is also kept as models/best.pt.
  • Press Ctrl+C for a clean exit — the current model is saved before quitting.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from agent import DQNAgent, TwentyFortyEightNet, DEVICE

# ─────────────────────────────── ANSI palette ─────────────────────────────────
_ESC = "\033["

def _c(code: str, text: str) -> str:
    return f"{_ESC}{code}m{text}{_ESC}0m"

# base colours
def dim(t):     return _c("2",       t)
def bold(t):    return _c("1",       t)
def red(t):     return _c("31",      t)
def green(t):   return _c("32",      t)
def yellow(t):  return _c("33",      t)
def blue(t):    return _c("34",      t)
def magenta(t): return _c("35",      t)
def cyan(t):    return _c("36",      t)
def white(t):   return _c("37",      t)
def bred(t):    return _c("1;31",    t)
def bgreen(t):  return _c("1;32",    t)
def byellow(t): return _c("1;33",    t)
def bblue(t):   return _c("1;34",    t)
def bmagenta(t):return _c("1;35",    t)
def bcyan(t):   return _c("1;36",    t)
def bwhite(t):  return _c("1;37",    t)

# tile value → color (mirrors the GUI palette conceptually)
def _tile_color(tile: int) -> str:
    if tile >= 2048: return bred(f"{tile:>5}")
    if tile >= 1024: return byellow(f"{tile:>5}")
    if tile >=  512: return yellow(f"{tile:>5}")
    if tile >=  256: return bgreen(f"{tile:>5}")
    if tile >=  128: return green(f"{tile:>5}")
    if tile >=   64: return cyan(f"{tile:>5}")
    return white(f"{tile:>5}")

# score coloring relative to rolling best
_session_best_score: int = 0
def _score_color(s: float) -> str:
    global _session_best_score
    if s >= _session_best_score:
        _session_best_score = int(s)
        return byellow(f"{int(s):>9,}")
    if s > _session_best_score * 0.75:
        return yellow(f"{int(s):>9,}")
    return white(f"{int(s):>9,}")


# ─────────────────────────────── layout helpers ────────────────────────────────
_COL_SEP = dim("  │  ")

HEADER = (
    bcyan(f"{'EP':>10}")
    + _COL_SEP
    + bwhite(f"{'AVG SCORE':>9}")
    + "  "
    + bwhite(f"{'MAX SCORE':>9}")
    + _COL_SEP
    + bblue(f"{'AVG MOVES':>9}")
    + _COL_SEP
    + byellow(f"{'MED TILE':>8}")
    + "  "
    + byellow(f"{'MAX TILE':>8}")
    + _COL_SEP
    + bmagenta(f"{'ε':>7}")
    + _COL_SEP
    + dim(f"{'BUF':>8}")
    + _COL_SEP
    + dim(f"{'LOSS':>8}")
)

_DIVIDER = dim("─" * 110)


def _fmt_row(
    ep_lo: int,
    ep_hi: int,
    scores: list[int],
    moves:  list[int],
    tiles:  list[int],
    eps:    float,
    buf:    int,
    loss:   float,
) -> str:
    ep_str  = cyan(f"{ep_lo:>6,}") + dim("-") + cyan(f"{ep_hi:<6,}")
    avg_sc  = _score_color(np.mean(scores))
    max_sc  = bwhite(f"{max(scores):>9,}")
    avg_mv  = bblue(f"{np.mean(moves):>9.1f}")
    med_til = _tile_color(int(np.median(tiles)))
    max_til = _tile_color(max(tiles))
    eps_str = magenta(f"{eps:>7.4f}")
    buf_str = dim(f"{buf:>8,}")
    loss_s  = dim(f"{loss:>8.4f}") if loss else dim(f"{'—':>8}")
    return (
        f"  {ep_str}"
        + _COL_SEP
        + avg_sc
        + "  "
        + max_sc
        + _COL_SEP
        + avg_mv
        + _COL_SEP
        + f"  {med_til}    {max_til}"
        + _COL_SEP
        + eps_str
        + _COL_SEP
        + buf_str
        + _COL_SEP
        + loss_s
    )


def _eval_banner(ep: int, stats: dict, elapsed_s: float) -> str:
    tile_bar = "  ".join(
        f"{_tile_color(t)}{dim('×')}{dim(str(n))}"
        for t, n in sorted(stats["tile_counts"].items(), reverse=True)[:8]
    )
    avg_score_s  = byellow(f'{stats["mean_score"]:>9,.0f}')
    max_score_s  = byellow(f'{stats["max_score"]:>9,}')
    games_s      = white(str(stats["n"]))
    avg_tile_s   = byellow(f'{stats["mean_tile"]:>8,.1f}')
    max_tile_s   = _tile_color(stats["max_tile"])
    elapsed_s2   = dim(f'{elapsed_s:.1f}s')
    lines = [
        "",
        bwhite("  ╔" + "═" * 72 + "╗"),
        bwhite("  ║") + bcyan(f"  EVALUATION  @  episode {ep:,}".center(72)) + bwhite("║"),
        bwhite("  ║") + " " * 72 + bwhite("║"),
        (bwhite("  ║") + f"  {bgreen('avg score')}: {avg_score_s}   "
                       + f"{bgreen('max score')}: {max_score_s}   "
                       + f"{bgreen('games')}: {games_s}".ljust(20)
                       + " " * 4 + bwhite("║")),
        (bwhite("  ║") + f"  {bgreen('avg tile')}: {avg_tile_s}   "
                       + f"{bgreen('max tile')}: {max_tile_s}   "
                       + f"  elapsed: {elapsed_s2}".ljust(24)
                       + " " * 4 + bwhite("║")),
        bwhite("  ║") + f"  {bgreen('tile dist')}: {tile_bar}" + bwhite("║"),
        bwhite("  ╚" + "═" * 72 + "╝"),
        "",
    ]
    return "\n".join(lines)


def _save_banner(path: str, ep: int) -> str:
    return (
        "  "
        + bgreen("✔")
        + dim(f" checkpoint saved → ")
        + cyan(path)
        + dim(f"  (ep {ep:,})")
    )


# ─────────────────────────────── training loop ────────────────────────────────
def train(
    n_episodes:  int  = 50_000,
    window:      int  = 25,      # print a stats row every N games
    eval_every:  int  = 500,
    eval_n:      int  = 50,
    save_every:  int  = 1_000,
    resume:      str | None = None,
    models_dir:  str  = "models",
):
    # ── setup ──────────────────────────────────────────────────────────────
    Path(models_dir).mkdir(exist_ok=True)

    print()
    print(bwhite("  ┌─────────────────────────────────────────────────────┐"))
    print(bwhite("  │") + bcyan("         2048  ·  Deep Q-Network  Trainer           ") + bwhite("│"))
    print(bwhite("  └─────────────────────────────────────────────────────┘"))
    print(f"  device      : {bgreen(str(DEVICE))}")
    net_params = sum(p.numel() for p in TwentyFortyEightNet().parameters())
    print(f"  parameters  : {bgreen(f'{net_params:,}')}")
    print(f"  episodes    : {bgreen(str(n_episodes))}")
    print(f"  models dir  : {bgreen(models_dir)}/")
    print()

    agent = DQNAgent()
    start_ep = 1

    if resume:
        agent.load(resume)
        start_ep = agent.episode_count + 1
        print(f"  {bgreen('resumed')} from {cyan(resume)}")
        print(f"  starting at episode {cyan(str(start_ep))}  ε={magenta(f'{agent.eps:.4f}')}")
        print()

    # ── per-window accumulators ─────────────────────────────────────────────
    w_scores: list[int]   = []
    w_moves:  list[int]   = []
    w_tiles:  list[int]   = []
    w_losses: list[float] = []
    w_start_ep            = start_ep

    best_eval_score: float = 0.0
    header_every          = 20   # re-print header every N rows

    row_count   = 0
    t_train_start = time.perf_counter()

    # ── graceful Ctrl-C ─────────────────────────────────────────────────────
    interrupted = False
    def _sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print()
        print(f"  {yellow('⚠')}  interrupt received — saving and exiting…")
    signal.signal(signal.SIGINT, _sigint)

    # ── main loop ───────────────────────────────────────────────────────────
    print(HEADER)
    print(_DIVIDER)

    for ep in range(start_ep, start_ep + n_episodes):
        if interrupted:
            break

        score, max_tile, steps = agent.run_episode(train=True)
        w_scores.append(score)
        w_moves.append(steps)
        w_tiles.append(max_tile)
        if agent.last_loss:
            w_losses.append(agent.last_loss)

        # ── print row ─────────────────────────────────────────────────────
        if len(w_scores) >= window:
            if row_count > 0 and row_count % header_every == 0:
                print()
                print(HEADER)
                print(_DIVIDER)

            print(_fmt_row(
                ep_lo   = w_start_ep,
                ep_hi   = ep,
                scores  = w_scores,
                moves   = w_moves,
                tiles   = w_tiles,
                eps     = agent.eps,
                buf     = len(agent.buffer),
                loss    = float(np.mean(w_losses)) if w_losses else 0.0,
            ))
            row_count   += 1
            w_start_ep   = ep + 1
            w_scores.clear()
            w_moves.clear()
            w_tiles.clear()
            w_losses.clear()

        # ── eval ──────────────────────────────────────────────────────────
        if ep % eval_every == 0:
            t0    = time.perf_counter()
            stats = agent.evaluate(eval_n)
            elapsed = time.perf_counter() - t0
            print(_eval_banner(ep, stats, elapsed))
            print(HEADER)
            print(_DIVIDER)
            row_count = 0

            # keep best model
            if stats["mean_score"] > best_eval_score:
                best_eval_score = stats["mean_score"]
                best_path = os.path.join(models_dir, "best.pt")
                agent.save(best_path)
                print(f"  {bred('★')} new best eval score {byellow(f'{best_eval_score:,.0f}')} → {cyan(best_path)}")

        # ── checkpoint ────────────────────────────────────────────────────
        if ep % save_every == 0:
            ckpt_path = os.path.join(models_dir, f"ckpt_ep{ep:06d}.pt")
            agent.save(ckpt_path)
            print(_save_banner(ckpt_path, ep))

    # ── final save ──────────────────────────────────────────────────────────
    final_path = os.path.join(models_dir, "final.pt")
    agent.save(final_path)
    total_time = time.perf_counter() - t_train_start
    h, m, s = int(total_time // 3600), int((total_time % 3600) // 60), int(total_time % 60)
    print()
    print(bwhite("  ─" * 40))
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
    p.add_argument("-e", "--episodes",   type=int, default=50_000,
                   metavar="N",   help="total training episodes (default: 50000)")
    p.add_argument("-r", "--resume",     type=str, default=None,
                   metavar="PATH", help="resume from checkpoint .pt file")
    p.add_argument("-w", "--window",     type=int, default=25,
                   metavar="N",   help="print stats every N games (default: 25)")
    p.add_argument("--eval-every",       type=int, default=500,
                   metavar="N",   help="run greedy evaluation every N episodes")
    p.add_argument("--eval-n",           type=int, default=50,
                   metavar="N",   help="number of greedy eval games per evaluation")
    p.add_argument("--save-every",       type=int, default=1_000,
                   metavar="N",   help="save checkpoint every N episodes")
    p.add_argument("--models-dir",       type=str, default="models",
                   metavar="DIR", help="directory for checkpoints (default: models/)")
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
    )
