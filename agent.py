"""
agent.py — Deep Q-Network agent for 2048.

Architecture
────────────
Input encoding:  (16, 4, 4) one-hot tensor.
                 Channel i is 1 wherever a cell holds value 2^i.
                 e.g. a "128" tile lights up channel 7 (2^7=128).

Network:         Conv2d(16 → 64,  kernel=2×2)
                 Conv2d(64 → 128, kernel=2×2)
                 Conv2d(128→ 128, kernel=2×2)
                 Linear(128 → 256) → Linear(256 → 4)

Algorithm:       Double DQN with a pre-allocated numpy replay buffer,
                 epsilon-greedy over *valid* actions only, configurable
                 reward shaping, and gradient clipping.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REWARD SHAPING  ←  edit RewardConfig
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All reward knobs live in the RewardConfig dataclass below.
Change the values there — no hunting through run_episode().

Key levers:
  survival_bonus   — small +reward every step the game continues
  log_scale        — apply log1p() to raw merge scores (keeps gradients sane)
  empty_weight     — bonus proportional to number of free cells (encourages
                     keeping the board open)
  monotone_weight  — bonus when rows/cols are monotonically sorted (classic
                     2048 heuristic: snaking tile order)
  merge_weight     — multiplier on the raw merge score component
  invalid_penalty  — negative reward for attempting an invalid move
                     (only fires if you remove valid-action filtering)
"""

from __future__ import annotations

import sys
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import Board

# ─────────────────────────────── globals ──────────────────────────────────────
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DIRS       = ["up", "down", "left", "right"]
N_CHANNELS = 16   # one-hot channels: log2 values 0…15  (covers up to tile 32768)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  REWARD SHAPING — edit these values to experiment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class RewardConfig:
    """
    Central place to tune every aspect of reward shaping.

    Quick-start recipes
    ───────────────────
    Vanilla (only merge score):
        merge_weight=1.0, survival_bonus=0.0, log_scale=True,
        empty_weight=0.0, monotone_weight=0.0

    Balanced (recommended default):
        merge_weight=1.0, survival_bonus=1.0, log_scale=True,
        empty_weight=0.1, monotone_weight=0.0

    Aggressive heuristics (good for faster early learning):
        merge_weight=1.0, survival_bonus=1.0, log_scale=True,
        empty_weight=0.3, monotone_weight=0.2
    """

    # ── merge reward ──────────────────────────────────────────────────────────
    merge_weight:  float = 1.0
    """Multiplier on the raw score gained from tile merges this step."""

    log_scale:     bool  = True
    """Apply log1p() to the total reward. Keeps Q-value magnitudes bounded
    when raw scores can spike to 65k+. Disable if you want raw rewards."""

    # ── step-level bonuses ────────────────────────────────────────────────────
    survival_bonus: float = 1.0
    """Flat bonus added every step the game is still alive.
    Motivates the agent to extend game length.
    Set to 0.0 to train purely on merge score."""

    empty_weight:  float = 0.1
    """Bonus = empty_weight × (number of empty cells after the move).
    Encourages keeping the board open so merges stay possible.
    Typical range: 0.0 – 0.5."""

    monotone_weight: float = 0.0
    """Bonus = monotone_weight × monotonicity_score(board).
    Rewards boards where each row/col is sorted (snake order).
    Classic 2048 heuristic. Start around 0.1–0.3 if you enable it."""

    # ── penalty ───────────────────────────────────────────────────────────────
    invalid_penalty: float = 0.0
    """Penalty for selecting an invalid action. Relevant only if you remove
    the valid-action filter in act(). Keep at 0.0 for normal training."""


# Module-level default — DQNAgent will use this unless you pass another one.
DEFAULT_REWARD_CFG = RewardConfig()


def _monotonicity(board_flat: np.ndarray) -> float:
    """
    Monotonicity heuristic: higher score when rows and columns are ordered
    (either ascending or descending). Uses log2 values so tile magnitudes
    are on a linear scale.

    Returns a value in [0, ~1] (normalised by max possible).
    """
    g = board_flat.reshape(4, 4).astype(np.float32)
    g = np.where(g > 0, np.log2(np.maximum(g, 1)), 0.0)
    score = 0.0
    for row in g:
        diff = np.diff(row)
        score += max(np.sum(diff[diff >= 0]), np.sum(-diff[diff <= 0]))
    for col in g.T:
        diff = np.diff(col)
        score += max(np.sum(diff[diff >= 0]), np.sum(-diff[diff <= 0]))
    return float(score) / 48.0   # 4 rows + 4 cols, max diff per pair ≈ 15


def compute_reward(
    prev_score: int,
    board: Board,
    cfg: RewardConfig,
) -> float:
    """
    Compute the shaped reward for one transition.

    Args:
        prev_score:  board.score *before* the move
        board:       Board object *after* the move (and tile spawn)
        cfg:         RewardConfig instance

    Returns:
        Scalar reward (float). Will be log1p-scaled if cfg.log_scale=True.
    """
    raw_merge = float(board.score - prev_score) * cfg.merge_weight

    reward = raw_merge + cfg.survival_bonus

    if cfg.empty_weight != 0.0:
        n_empty = int(np.sum(board.board == 0))
        reward += cfg.empty_weight * n_empty

    if cfg.monotone_weight != 0.0:
        reward += cfg.monotone_weight * _monotonicity(board.board)

    if cfg.log_scale:
        # log1p on the positive part; preserve sign for any negative reward
        reward = float(np.sign(reward) * np.log1p(abs(reward)))

    return reward


# ─────────────────────────────── encoding ─────────────────────────────────────
def encode(flat: np.ndarray) -> np.ndarray:
    """
    flat: int32 array of shape (16,) — raw board values.
    Returns: float32 array of shape (16, 4, 4) — one-hot log2 encoding.
    """
    out  = np.zeros((N_CHANNELS, 4, 4), dtype=np.float32)
    mask = flat > 0
    log2 = np.zeros(16, dtype=np.int64)
    log2[mask] = np.log2(flat[mask]).astype(np.int64)
    np.clip(log2, 0, N_CHANNELS - 1, out=log2)
    rows = np.arange(16) // 4
    cols = np.arange(16) % 4
    out[log2, rows, cols] = 1.0
    return out


# ─────────────────────────────── network ──────────────────────────────────────
class TwoZeroFourEightNet(nn.Module):
    """
    Three stacked 2×2 convolutions → fully connected head.

    Spatial walkthrough (no padding):
      Input      : (B, 16, 4, 4)
      After conv1: (B, 64,  3, 3)
      After conv2: (B, 128, 2, 2)
      After conv3: (B, 128, 1, 1)
      Flatten    : (B, 128)
      FC1        : (B, 256)
      FC2 (out)  : (B, 4)    ← Q-value per action
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(N_CHANNELS, 64,  kernel_size=2)
        self.conv2 = nn.Conv2d(64,         128, kernel_size=2)
        self.conv3 = nn.Conv2d(128,        128, kernel_size=2)
        self.fc1   = nn.Linear(128, 256)
        self.fc2   = nn.Linear(256, 4)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ─────────────────────────────── replay buffer ────────────────────────────────
class ReplayBuffer:
    """
    Pre-allocated circular numpy buffer.  Much faster than a Python deque of
    tensors because we do one big numpy fancy-index on every sample() call.
    """

    def __init__(self, capacity: int = 100_000):
        self.cap   = capacity
        self.ptr   = 0
        self.size  = 0

        self.states  = np.zeros((capacity, N_CHANNELS, 4, 4), dtype=np.float32)
        self.nstates = np.zeros((capacity, N_CHANNELS, 4, 4), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.bool_)

    def push(self, state, action, reward, nstate, done):
        i = self.ptr
        self.states[i]  = state
        self.nstates[i] = nstate
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i]   = done
        self.ptr  = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int):
        idx = np.random.randint(0, self.size, batch)
        s  = torch.from_numpy(self.states[idx]).to(DEVICE)
        a  = torch.from_numpy(self.actions[idx]).to(DEVICE)
        r  = torch.from_numpy(self.rewards[idx]).to(DEVICE)
        ns = torch.from_numpy(self.nstates[idx]).to(DEVICE)
        d  = torch.from_numpy(self.dones[idx]).to(DEVICE)
        return s, a, r, ns, d

    def __len__(self) -> int:
        return self.size


# ─────────────────────────────── agent ────────────────────────────────────────
class DQNAgent:
    """
    Double DQN agent.

    Pass a custom RewardConfig to __init__ to change reward shaping:

        cfg = RewardConfig(survival_bonus=0.5, empty_weight=0.2, monotone_weight=0.1)
        agent = DQNAgent(reward_cfg=cfg)
    """

    def __init__(
        self,
        lr:          float = 1e-4,
        gamma:       float = 0.99,
        eps_start:   float = 1.0,
        eps_end:     float = 0.05,
        eps_decay:   float = 0.9998,
        batch_size:  int   = 512,
        buffer_cap:  int   = 100_000,
        target_sync: int   = 500,
        warmup:      int   = 2_000,
        # ── reward shaping ────────────────────────────────────────────────────
        reward_cfg:  RewardConfig | None = None,
        # ── performance ───────────────────────────────────────────────────────
        learn_every: int   = 1,   # run learn() every N environment steps
                                  # set to 2 or 4 to halve/quarter gradient cost
    ):
        self.gamma       = gamma
        self.eps         = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.batch_size  = batch_size
        self.target_sync = target_sync
        self.warmup      = warmup
        self.learn_every = learn_every
        self.reward_cfg  = reward_cfg or DEFAULT_REWARD_CFG

        self.policy = TwoZeroFourEightNet().to(DEVICE)
        self.target = TwoZeroFourEightNet().to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optim  = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)
        self.last_loss:     float = 0.0
        self.episode_count: int   = 0
        self._step_count:   int   = 0   # total env steps for learn_every

    # ── action selection ────────────────────────────────────────────────────
    @torch.no_grad()
    def act(self, state_np: np.ndarray, valid: list[int]) -> int:
        if not valid:
            return 0
        if random.random() < self.eps:
            return random.choice(valid)
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)
        q = self.policy(state_t)[0]
        mask = torch.full((4,), float("-inf"), device=DEVICE)
        for a in valid:
            mask[a] = q[a]
        return int(mask.argmax())

    # ── learning step ────────────────────────────────────────────────────────
    def learn(self) -> float | None:
        if len(self.buffer) < max(self.batch_size, self.warmup):
            return None

        s, a, r, ns, done = self.buffer.sample(self.batch_size)

        # Double DQN: policy picks action, target evaluates it
        q_pred = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_next_a = self.policy(ns).argmax(1, keepdim=True)
            q_next      = self.target(ns).gather(1, best_next_a).squeeze(1)
            q_target    = r + self.gamma * q_next * (~done)

        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optim.step()

        self.grad_steps = getattr(self, "grad_steps", 0) + 1
        if self.grad_steps % self.target_sync == 0:
            self.target.load_state_dict(self.policy.state_dict())

        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.last_loss = float(loss.detach().item())
        return self.last_loss

    # ── episode runner ───────────────────────────────────────────────────────
    def run_episode(self, train: bool = True) -> tuple[int, int, int]:
        """
        Play one complete game.

        Returns:
            (final_score, max_tile_value, number_of_steps)
        """
        board  = Board()
        state  = encode(board.board)
        steps  = 0

        while not board.game_over:
            valid  = board.valid_actions()
            action = self.act(state, valid) if train else self._greedy(state, valid)

            prev_score = board.score
            board.move(DIRS[action])

            next_state = encode(board.board)
            done       = board.game_over

            if train:
                reward = compute_reward(prev_score, board, self.reward_cfg)
                self.buffer.push(state, action, reward, next_state, done)
                self._step_count += 1
                if self._step_count % self.learn_every == 0:
                    self.learn()

            state  = next_state
            steps += 1

        self.episode_count += 1
        return board.score, int(board.board.max()), steps

    @torch.no_grad()
    def _greedy(self, state_np: np.ndarray, valid: list[int]) -> int:
        if not valid:
            return 0
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)
        q = self.policy(state_t)[0]
        mask = torch.full((4,), float("-inf"), device=DEVICE)
        for a in valid:
            mask[a] = q[a]
        return int(mask.argmax())

    # ── evaluate ─────────────────────────────────────────────────────────────
    def evaluate(self, n: int = 100) -> dict:
        scores, tiles = [], []
        for _ in range(n):
            s, t, _ = self.run_episode(train=False)
            scores.append(s)
            tiles.append(t)
        tile_counts = {}
        for t in tiles:
            tile_counts[t] = tile_counts.get(t, 0) + 1
        return {
            "mean_score":  float(np.mean(scores)),
            "max_score":   int(max(scores)),
            "mean_tile":   float(np.mean(tiles)),
            "max_tile":    int(max(tiles)),
            "tile_counts": dict(sorted(tile_counts.items())),
            "n":           n,
        }

    # ── checkpoint ────────────────────────────────────────────────────────────
    def save(self, path: str = "agent.pt"):
        torch.save(
            {
                "policy":        self.policy.state_dict(),
                "target":        self.target.state_dict(),
                "optim":         self.optim.state_dict(),
                "eps":           self.eps,
                "grad_steps":    getattr(self, "grad_steps", 0),
                "episode_count": self.episode_count,
                "reward_cfg":    self.reward_cfg,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.optim.load_state_dict(ckpt["optim"])
        self.eps            = ckpt["eps"]
        self.grad_steps     = ckpt.get("grad_steps", 0)
        self.episode_count  = ckpt.get("episode_count", 0)
        if "reward_cfg" in ckpt:
            self.reward_cfg = ckpt["reward_cfg"]


# ─────────────────────────────── entry point ──────────────────────────────────
if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    from train import train # type: ignore
    train(resume=ckpt)