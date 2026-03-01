"""
train.py — curriculum DQN vs Minimax opponent.

Training:  every episode starts from a random mid-game position (4-8 moves
           already played) so the agent sees meaningful board states even
           during early high-epsilon exploration.

Promotion: every EVAL_INTERVAL episodes, play EVAL_GAMES fully greedy (ε=0)
           games from random positions. Need EVAL_WIN_THRESHOLD wins to
           advance to the next minimax depth. No rolling-window noise.

Speed fixes:
  - BUFFER_CAPACITY reduced to 20k (avoids MPS memory pressure spike)
  - Training games use FastBoard throughout; only one Board.board_to_tensor
    call per agent move (unavoidable for inference)
  - Eval runs entirely on FastBoard — zero tensor ops on opponent side
"""

import os
import time
import random
import statistics

import torch

from board import Board
from agent import DQNAgent
from minimax import MinimaxOpponent, FastBoard, ROWS, COLS, SIZE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAVE_DIR        = "models"
RUN_NAME        = "cf_dqn"
NUM_EPISODES    = 50_000
SAVE_INTERVAL   = 5_000
WINDOW          = 250          # print stats every N training episodes

# Curriculum
MAX_DEPTH       = 6

# Training random starts
TRAIN_RAND_MIN  = 4
TRAIN_RAND_MAX  = 8

# Promotion eval
EVAL_INTERVAL       = 1_000
EVAL_GAMES          = 100
EVAL_WIN_THRESHOLD  = 60       # 60/100 = 60%
EVAL_RAND_MIN       = 4
EVAL_RAND_MAX       = 20

# DQN hyper-params
LR              = 2e-3
GAMMA           = 0.99
BATCH_SIZE      = 64
BUFFER_CAPACITY = 20_000       # reduced from 60k — avoids MPS memory pressure
TARGET_UPDATE   = 500
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 12_000

# Rewards
WIN_REWARD  =  1.0
LOSS_REWARD = -1.0
DRAW_REWARD =  0.2

RESUME_PATH = ""

# ---------------------------------------------------------------------------

class ANSI:
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    RESET   = "\033[0m"


# ---------------------------------------------------------------------------
# FastBoard helpers
# ---------------------------------------------------------------------------

def _random_fastboard(min_moves: int, max_moves: int) -> "FastBoard":
    """Non-terminal FastBoard with min_moves..max_moves already played."""
    while True:
        fb = FastBoard()
        n  = random.randint(min_moves, max_moves)
        terminal = False
        for _ in range(n):
            valid = [c for c in range(COLS) if fb.heights[c] < ROWS]
            if not valid:
                terminal = True
                break
            col = random.choice(valid)
            r   = fb.make_move(col)
            if fb.check_win_at(r * COLS + col, fb.last_player()):
                terminal = True
                break
        if not terminal:
            return fb


def _fb_to_board(fb: FastBoard) -> Board:
    """FastBoard → tensor Board for agent inference only."""
    board = Board()
    cells = fb.cells
    for r in range(ROWS):
        for c in range(COLS):
            v = cells[r * COLS + c]
            if v == 1:
                board.player1_bits[r, c] = 1.0
            elif v == 2:
                board.player2_bits[r, c] = 1.0
    board.turn = fb.turn
    return board


# ---------------------------------------------------------------------------
# Training episode
# ---------------------------------------------------------------------------

def play_episode(agent: DQNAgent, opponent: MinimaxOpponent, agent_is_p1: bool) -> tuple[int, int]:
    fb           = _random_fastboard(TRAIN_RAND_MIN, TRAIN_RAND_MAX)
    agent_player = 1 if agent_is_p1 else 2
    opp_player   = 3 - agent_player
    trajectory   = []
    start_turn   = fb.turn

    while True:
        valid = [c for c in range(COLS) if fb.heights[c] < ROWS]
        if not valid:
            winner = 0
            break

        current = (fb.turn & 1) + 1

        if current == agent_player:
            board  = _fb_to_board(fb)
            state  = Board.board_to_tensor(board)
            action = agent.select_action(board)
        else:
            action = opponent.select_action_fast(fb)

        r = fb.make_move(action)
        if r < 0:
            winner = opp_player
            break

        cell = r * COLS + action
        last = fb.last_player()

        if fb.check_win_at(cell, last):
            winner = last
            if current == agent_player:
                ns = Board.board_to_tensor(_fb_to_board(fb))
                trajectory.append((state, action, WIN_REWARD, ns, True)) # type: ignore
            break

        if fb.is_full():
            winner = 0
            if current == agent_player:
                ns = Board.board_to_tensor(_fb_to_board(fb))
                trajectory.append((state, action, DRAW_REWARD, ns, True)) # type: ignore
            break

        if current == agent_player:
            ns = Board.board_to_tensor(_fb_to_board(fb))
            trajectory.append((state, action, 0.0, ns, False)) # type: ignore

    if trajectory and winner == opp_player:
        s, a, _, ns, d = trajectory[-1]
        trajectory[-1] = (s, a, LOSS_REWARD, ns, True)

    for t in trajectory:
        agent.buffer.push(*t)
    agent.learn()

    return winner, fb.turn - start_turn


# ---------------------------------------------------------------------------
# Promotion eval — greedy, wider random starts, no learning
# ---------------------------------------------------------------------------

def run_eval(agent: DQNAgent, opponent: MinimaxOpponent) -> tuple[int, int, int]:
    wins = draws = losses = 0
    for i in range(EVAL_GAMES):
        fb           = _random_fastboard(EVAL_RAND_MIN, EVAL_RAND_MAX)
        agent_player = 1 if (i % 2 == 0) else 2

        while True:
            valid = [c for c in range(COLS) if fb.heights[c] < ROWS]
            if not valid:
                winner = 0
                break
            current = (fb.turn & 1) + 1
            if current == agent_player:
                action = agent.policy_net.best_move(_fb_to_board(fb))
            else:
                action = opponent.select_action_fast(fb)
            r = fb.make_move(action)
            if r < 0:
                winner = 3 - agent_player
                break
            cell = r * COLS + action
            last = fb.last_player()
            if fb.check_win_at(cell, last):
                winner = last
                break
            if fb.is_full():
                winner = 0
                break

        if winner == agent_player:   wins   += 1
        elif winner == 0:            draws  += 1
        else:                        losses += 1

    return wins, draws, losses


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(agent: DQNAgent, episode: int, depth: int) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{RUN_NAME}_ep{episode}_depth{depth}.pth")
    agent.save(path)
    return path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_training() -> None:
    print(f"Curriculum DQN — {NUM_EPISODES} episodes | Minimax depth 1→{MAX_DEPTH}")
    print(f"Train starts: random {TRAIN_RAND_MIN}–{TRAIN_RAND_MAX} moves in")
    print(f"Promotion: every {EVAL_INTERVAL} eps | "
          f"{EVAL_WIN_THRESHOLD}/{EVAL_GAMES} greedy wins from "
          f"random {EVAL_RAND_MIN}–{EVAL_RAND_MAX}-move positions\n")

    current_depth = 1
    opponent      = MinimaxOpponent(depth=current_depth)

    agent = DQNAgent(
        lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE,
        epsilon_start=EPS_START, epsilon_end=EPS_END, epsilon_decay=EPS_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
    )

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        agent.load(RESUME_PATH)
        print(f"Resumed from {RESUME_PATH}")

    w_wins = w_losses = w_draws = 0
    game_lengths = []
    t0 = time.perf_counter()

    for ep in range(1, NUM_EPISODES + 1):
        agent_is_p1 = (ep % 2 == 1)
        winner, length = play_episode(agent, opponent, agent_is_p1)
        game_lengths.append(length)

        ap = 1 if agent_is_p1 else 2
        if winner == ap:   w_wins   += 1
        elif winner == 0:  w_draws  += 1
        else:              w_losses += 1

        if ep % WINDOW == 0:
            total   = w_wins + w_losses + w_draws or 1
            avg_len = statistics.mean(game_lengths[-WINDOW:])
            wr      = w_wins / total
            print(
                f"Ep {ep:>6} | Depth {ANSI.CYAN}{current_depth}{ANSI.RESET} | "
                f"W {ANSI.GREEN}{w_wins:>3}{ANSI.RESET} "
                f"L {ANSI.RED}{w_losses:>3}{ANSI.RESET} "
                f"D {ANSI.CYAN}{w_draws:>2}{ANSI.RESET} "
                f"/{WINDOW} "
                f"({ANSI.YELLOW}{wr:.1%} ε-train{ANSI.RESET}) | "
                f"AvgLen {avg_len:>4.1f} | "
                f"ε {ANSI.MAGENTA}{agent.epsilon:.3f}{ANSI.RESET} | "
                f"{time.perf_counter()-t0:>6.1f}s"
            )
            w_wins = w_losses = w_draws = 0

        if ep % EVAL_INTERVAL == 0 and current_depth < MAX_DEPTH:
            ew, ed, el = run_eval(agent, opponent)
            promoted   = ew >= EVAL_WIN_THRESHOLD
            tag = (f"{ANSI.GREEN}✓ PROMOTE → depth {current_depth+1}{ANSI.RESET}"
                   if promoted else
                   f"{ANSI.RED}not yet ({ew}/{EVAL_WIN_THRESHOLD}){ANSI.RESET}")
            print(
                f"  EVAL depth {ANSI.CYAN}{current_depth}{ANSI.RESET} | "
                f"W {ANSI.GREEN}{ew}{ANSI.RESET} "
                f"D {ANSI.CYAN}{ed}{ANSI.RESET} "
                f"L {ANSI.RED}{el}{ANSI.RESET} "
                f"/{EVAL_GAMES} greedy → {tag}"
            )
            if promoted:
                save_checkpoint(agent, ep, current_depth)
                current_depth += 1
                opponent       = MinimaxOpponent(depth=current_depth)
                game_lengths.clear()
                print()

        if ep % SAVE_INTERVAL == 0:
            print(f"  → Checkpoint: {save_checkpoint(agent, ep, current_depth)}")

    agent.save(os.path.join(SAVE_DIR, f"{RUN_NAME}_final.pth"))
    print(f"\nDone. {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run_training()