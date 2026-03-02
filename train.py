"""
train.py — curriculum DQN vs Minimax opponent with reward shaping.

Key changes vs previous version:
  - Reward shaping: agent gets intermediate signal every move, not just
    win/loss at the end. Shaped rewards teach it WHAT to do, not just
    whether it eventually won.
  - Training starts from empty board (no random offset) so agent learns
    full game from the start, including openings.
  - Eval still uses random start positions for diversity.
  - Curriculum stays: depth 0 (random) → depth 1 → ... → depth 6
"""

import os
import time
import random
import statistics

import torch

from board import Board
from agent import DQNAgent
from minimax import MinimaxOpponent, FastBoard, ROWS, COLS, SIZE, _WIN_LINES, _CELL_WIN_LINES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAVE_DIR        = "models"
RUN_NAME        = "cf_dqn"
NUM_EPISODES    = 100_000
SAVE_INTERVAL   = 5_000
WINDOW          = 250

MAX_DEPTH       = 6   # depth 0=random, 1-6=minimax

# Eval
EVAL_INTERVAL       = 1_000
EVAL_GAMES          = 100
EVAL_WIN_THRESHOLD  = 60
EVAL_RAND_MIN       = 0    # eval from empty board — matches training distribution
EVAL_RAND_MAX       = 8

# DQN
LR              = 1e-3
GAMMA           = 0.99
BATCH_SIZE      = 64
BUFFER_CAPACITY = 100_000
LEARN_EVERY = 4
TARGET_UPDATE   = 200
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 5_000    # was 15_000 — reach near-greedy by ep ~15k

# Terminal rewards
WIN_REWARD  =  1.0
LOSS_REWARD = -1.0
DRAW_REWARD =  0.3

# Shaped intermediate rewards — teach strategy every move
THREAT_4_REWARD     =  0.6   # agent creates a 3-in-a-row with open end (one move from win)
OPP_THREAT_4_REWARD = -0.6   # agent lets opponent create same
THREAT_3_REWARD     =  0.15  # agent creates a 2-in-a-row with open ends
OPP_THREAT_3_REWARD = -0.15  # opponent creates same
CENTER_REWARD       =  0.03  # agent plays in centre column (col 3)

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
# Reward shaping helpers — all on FastBoard, pure Python
# ---------------------------------------------------------------------------

def _count_threats(cells: bytearray, player: int, n: int) -> int:
    """
    Count win lines with exactly n pieces of `player` and 0 opponent pieces.
    n=3 → threat-4 (one move from winning)
    n=2 → threat-3 (two moves from winning)
    """
    opp   = 3 - player
    count = 0
    for a, b, c, d in _WIN_LINES:
        ca, cb, cc, cd = cells[a], cells[b], cells[c], cells[d]
        if ca == opp or cb == opp or cc == opp or cd == opp:
            continue
        if (ca == player) + (cb == player) + (cc == player) + (cd == player) == n:
            count += 1
    return count


def _shaped_reward(cells_before: bytearray, cells_after: bytearray,
                   agent: int, col: int) -> float:
    """Intermediate reward based on how the move changed the threat landscape."""
    opp = 3 - agent

    new_agent_4 = _count_threats(cells_after, agent, 3) - _count_threats(cells_before, agent, 3)
    new_agent_3 = _count_threats(cells_after, agent, 2) - _count_threats(cells_before, agent, 2)
    new_opp_4   = max(0, _count_threats(cells_after, opp, 3) - _count_threats(cells_before, opp, 3))
    new_opp_3   = max(0, _count_threats(cells_after, opp, 2) - _count_threats(cells_before, opp, 2))

    r  = new_agent_4 * THREAT_4_REWARD
    r += new_agent_3 * THREAT_3_REWARD
    r -= new_opp_4   * abs(OPP_THREAT_4_REWARD)
    r -= new_opp_3   * abs(OPP_THREAT_3_REWARD)
    r += CENTER_REWARD if col == 3 else 0.0
    return float(r)


# ---------------------------------------------------------------------------
# FastBoard helpers
# ---------------------------------------------------------------------------

def _random_fastboard(min_moves: int, max_moves: int) -> "FastBoard":
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


def _fb_to_board(fb: "FastBoard") -> Board:
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
# Training episode — full games from empty board, shaped rewards
# ---------------------------------------------------------------------------

def play_episode(agent: DQNAgent, opponent: MinimaxOpponent,
                 agent_is_p1: bool) -> tuple[int, int]:
    fb           = FastBoard()          # always start from empty board
    agent_player = 1 if agent_is_p1 else 2
    opp_player   = 3 - agent_player
    trajectory   = []

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
            cells_before = bytearray(fb.cells)   # snapshot for shaping
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
            # Shaped intermediate reward
            reward = _shaped_reward(cells_before, fb.cells, agent_player, action) # type: ignore
            ns = Board.board_to_tensor(_fb_to_board(fb))
            trajectory.append((state, action, reward, ns, False)) # type: ignore
    # Back-fill terminal loss reward
    if trajectory and winner == opp_player:
        s, a, _, ns, d = trajectory[-1]
        trajectory[-1] = (s, a, LOSS_REWARD, ns, True)

    for t in trajectory:
        agent.buffer.push(*t)
    # Learning is now performed externally every LEARN_EVERY episodes.

    return winner, fb.turn


# ---------------------------------------------------------------------------
# Promotion eval — greedy, random starts, no learning
# ---------------------------------------------------------------------------

def run_eval(agent: DQNAgent, opponent: MinimaxOpponent) -> tuple[int, int, int]:
    """
    Greedy (ε=0) games. Half start from empty board, half from 2-8 move
    random positions — same distribution the agent trained on.
    """
    wins = draws = losses = 0
    for i in range(EVAL_GAMES):
        # Match training distribution: empty board or short random start
        if i % 2 == 0:
            fb = FastBoard()
        else:
            fb = _random_fastboard(2, 8)
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
    # Save model + optimizer + step counter + current curriculum depth
    torch.save({
        "policy_net": agent.policy_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "steps_done": agent.steps_done,
        "current_depth": int(depth),
    }, path)
    print(f"[Agent] Saved → {path}")
    return path


def find_latest_checkpoint(save_dir: str) -> str | None:
    """Return the newest .pt/.pth checkpoint path in `save_dir`, or None if none found."""
    if not os.path.isdir(save_dir):
        return None
    candidates = []
    for name in os.listdir(save_dir):
        if name.endswith(".pt") or name.endswith(".pth"):
            candidates.append(os.path.join(save_dir, name))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_training() -> None:
    print(f"Curriculum DQN — {NUM_EPISODES} episodes | depth 0 (random) → {MAX_DEPTH} (minimax)")
    print(f"Reward shaping: ON  (threat-4: ±{THREAT_4_REWARD}, threat-3: ±{THREAT_3_REWARD}, centre: +{CENTER_REWARD})")
    print(f"Promotion: every {EVAL_INTERVAL} eps | {EVAL_WIN_THRESHOLD}/{EVAL_GAMES} greedy wins\n")

    current_depth = 0
    opponent      = MinimaxOpponent(depth=current_depth)

    agent = DQNAgent(
        lr=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE,
        epsilon_start=EPS_START, epsilon_end=EPS_END, epsilon_decay=EPS_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
    )

    # Resume from explicit path, or if none provided, attempt to load the
    # latest checkpoint found in `SAVE_DIR` for convenience.
    resume_path = RESUME_PATH
    if not resume_path:
        latest = find_latest_checkpoint(SAVE_DIR)
        if latest:
            resume_path = latest
            print(f"Auto-resume: found latest checkpoint {resume_path}")
    if resume_path and os.path.exists(resume_path):
        # Load checkpoint payload to recover curriculum depth, then load weights
        ckpt = torch.load(resume_path, map_location="cpu")
        agent.load(resume_path)
        current_depth = int(ckpt.get("current_depth", 0))
        opponent = MinimaxOpponent(depth=current_depth)
        print(f"Resumed from {resume_path} at depth {current_depth}")

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
            total      = w_wins + w_losses + w_draws or 1
            avg_len    = statistics.mean(game_lengths[-WINDOW:])
            wr         = w_wins / total
            depth_lbl  = "rng" if current_depth == 0 else str(current_depth)
            print(
                f"Ep {ep:>6} | Opp {ANSI.CYAN}{depth_lbl:>3}{ANSI.RESET} | "
                f"W {ANSI.GREEN}{w_wins:>3}{ANSI.RESET} "
                f"L {ANSI.RED}{w_losses:>3}{ANSI.RESET} "
                f"D {ANSI.CYAN}{w_draws:>2}{ANSI.RESET} "
                f"/{WINDOW} "
                f"({ANSI.GREEN if wr >= 0.5 else ANSI.YELLOW}{wr:.1%}{ANSI.RESET}) | "
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
            depth_lbl = "random" if current_depth == 0 else str(current_depth)
            print(
                f"  EVAL opp={depth_lbl} | "
                f"W {ANSI.GREEN}{ew}{ANSI.RESET} "
                f"D {ANSI.CYAN}{ed}{ANSI.RESET} "
                f"L {ANSI.RED}{el}{ANSI.RESET} "
                f"→ {tag}"
            )
            if promoted:
                save_checkpoint(agent, ep, current_depth)
                current_depth += 1
                opponent       = MinimaxOpponent(depth=current_depth)
                game_lengths.clear()
                # Promotion reset: bump exploration so buffer refills with diverse
                # experiences against the new opponent before going greedy.
                try:
                    agent.start_epsilon_bump(0.25, decay_steps=2000)
                except Exception:
                    pass
                print()

        if ep % SAVE_INTERVAL == 0:
            print(f"  → Checkpoint: {save_checkpoint(agent, ep, current_depth)}")

        # Perform learning less frequently to reduce catastrophic forgetting
        if ep % LEARN_EVERY == 0:
            _ = agent.learn()

    agent.save(os.path.join(SAVE_DIR, f"{RUN_NAME}_final.pth"))
    print(f"\nDone. {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run_training()