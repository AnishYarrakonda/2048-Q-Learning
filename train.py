"""
train.py — curriculum DQN, rolling-window promotion (no separate eval).

Promotion: after every 250-game window, if the agent won >= 60% of those
games (>= 150/250) it advances to the next Minimax depth.
Stats and promotion check happen together at the same print interval.
"""

import os
import time
import statistics

import torch

from board import Board
from agent import DQNAgent
from minimax import MinimaxOpponent as MCTSOpponent, FastBoard

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAVE_DIR        = "models"
RUN_NAME        = "cf_dqn"
NUM_EPISODES    = 50_000
SAVE_INTERVAL   = 5_000
WINDOW          = 250             # print + promotion check every N episodes

# Curriculum
MAX_MCTS_DEPTH  = 5   # depth goes 1..6 (stage+1)


# Promotion — 60% wins in the last WINDOW games (epsilon-inclusive)
PROMOTE_THRESHOLD = 0.60          # 150 / 250

# DQN hyper-params
LR              = 2e-3
GAMMA           = 0.99
BATCH_SIZE      = 64
BUFFER_CAPACITY = 60_000
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
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    MAGENTA= "\033[95m"
    RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Training episode — entirely on FastBoard except agent inference
# ---------------------------------------------------------------------------

def play_episode(agent: DQNAgent, opponent: MCTSOpponent, agent_is_p1: bool) -> tuple[int, int]:
    board        = Board()
    agent_player = 1 if agent_is_p1 else 2
    opp_player   = 3 - agent_player
    trajectory   = []

    while True:
        current = (board.turn & 1) + 1
        valid   = board.valid_moves()
        if not valid:
            winner = 0
            break

        state  = Board.board_to_tensor(board)

        if current == agent_player:
            action = agent.select_action(board)
        else:
            # Opponent works on FastBoard — convert once per opponent move
            action = opponent.select_action(board)

        row = board.make_move(action)
        if row is None:
            winner = opp_player
            break

        done, winner = board.game_over(row, action)

        if current == agent_player:
            if done:
                reward = WIN_REWARD if winner == agent_player else (DRAW_REWARD if winner == 0 else LOSS_REWARD)
            else:
                reward = 0.0
            trajectory.append((state, action, reward, Board.board_to_tensor(board), done))

        if done:
            break

    if trajectory and winner == opp_player:
        s, a, _, ns, d = trajectory[-1]
        trajectory[-1] = (s, a, LOSS_REWARD, ns, True)

    for t in trajectory:
        agent.buffer.push(*t)
    agent.learn()

    return winner, board.turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_checkpoint(agent: DQNAgent, episode: int, stage: int) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{RUN_NAME}_ep{episode}_stage{stage}.pth")
    agent.save(path)
    return path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_training() -> None:
    win_threshold = int(PROMOTE_THRESHOLD * WINDOW)   # 150
    print(f"Curriculum DQN — {NUM_EPISODES} episodes | Minimax depth 1→6")
    print(f"Promotion: {win_threshold}/{WINDOW} wins in rolling window (ε-inclusive)\n")

    agent = DQNAgent(
        lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE,
        epsilon_start=EPS_START, epsilon_end=EPS_END, epsilon_decay=EPS_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
    )

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        agent.load(RESUME_PATH)
        print(f"Resumed from {RESUME_PATH}")

    stage    = 0
    opponent = MCTSOpponent(depth=stage + 1)

    w_wins = w_losses = w_draws = 0
    game_lengths = []
    t0 = time.perf_counter()

    for ep in range(1, NUM_EPISODES + 1):
        agent_is_p1 = (ep % 2 == 1)
        winner, length = play_episode(agent, opponent, agent_is_p1)
        game_lengths.append(length)

        ap = (1 if agent_is_p1 else 2)
        if winner == ap:   w_wins   += 1
        elif winner == 0:  w_draws  += 1
        else:              w_losses += 1

        if ep % WINDOW == 0:
            total   = w_wins + w_losses + w_draws or 1
            avg_len = statistics.mean(game_lengths[-WINDOW:])
            wr      = w_wins / total
            promote = (wr >= PROMOTE_THRESHOLD) and (stage < MAX_MCTS_DEPTH)

            promo_str = ""
            if promote:
                save_checkpoint(agent, ep, stage)
                stage   += 1
                opponent = MCTSOpponent(depth=stage + 1)
                promo_str = f" {ANSI.GREEN}→ PROMOTED depth {stage + 1}{ANSI.RESET}"

            print(
                f"Ep {ep:>6} | Depth {ANSI.CYAN}{(stage - (1 if promote else 0)) + 1}{ANSI.RESET} | "
                f"W {ANSI.GREEN}{w_wins}{ANSI.RESET} "
                f"L {ANSI.RED}{w_losses}{ANSI.RESET} "
                f"D {ANSI.CYAN}{w_draws}{ANSI.RESET} "
                f"/{WINDOW} "
                f"({ANSI.GREEN if wr >= PROMOTE_THRESHOLD else ANSI.YELLOW}{wr:.1%}{ANSI.RESET}) | "
                f"AvgLen {avg_len:>4.1f} | "
                f"ε {ANSI.MAGENTA}{agent.epsilon:.3f}{ANSI.RESET} | "
                f"{time.perf_counter()-t0:>6.1f}s"
                f"{promo_str}"
            )

            w_wins = w_losses = w_draws = 0

        if ep % SAVE_INTERVAL == 0:
            print(f"  → Checkpoint: {save_checkpoint(agent, ep, stage)}")

    agent.save(os.path.join(SAVE_DIR, f"{RUN_NAME}_final.pth"))
    print(f"\nDone. {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run_training()