"""
agent.py — Deep Q-Network agent for 2048.

Architecture
────────────
Input encoding:  (16, 4, 4) one-hot tensor.
                 Channel i is 1 wherever a cell holds value 2^i.
                 e.g. a "128" tile lights up channel 7 (2^7=128).
                 Using log2 one-hot instead of raw values stops the loss
                 from being dominated by large tile magnitudes and gives
                 the network a distinct learned feature per power-of-two.

Network:         Conv2d(16 → 64,  kernel=2×2)   ← the 64 2×2 kernels
                 Conv2d(64 → 128, kernel=2×2)
                 Conv2d(128→ 128, kernel=2×2)
                 Linear(128 → 256) → Linear(256 → 4)

                 Three stacked 2×2 convolutions on a 4×4 board collapse
                 spatial dims to 1×1 while each kernel captures every
                 possible adjacent-pair pattern (the key relationship in 2048).

Algorithm:       Double DQN with a pre-allocated numpy replay buffer,
                 epsilon-greedy over *valid* actions only, log1p reward
                 scaling, and gradient clipping.

Usage
─────
  # Train from scratch:
  python agent.py

  # Resume from checkpoint:
  python agent.py agent_ep5000.pt

  # Import and use:
  from agent import DQNAgent, DIRS
  agent = DQNAgent()
  agent.load("agent_final.pt")
  score, max_tile, _ = agent.run_episode(train=False)
"""

from __future__ import annotations

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import Board

# ─────────────────────────────── globals ──────────────────────────────────────
DEVICE     = torch.device("mps" if torch.mps.is_available() else "cpu")
DIRS       = ["up", "down", "left", "right"]
N_CHANNELS = 16   # one-hot channels: log2 values 0 … 15  (covers up to tile 32768)


# ─────────────────────────────── encoding ─────────────────────────────────────
def encode(flat: np.ndarray) -> np.ndarray:
    """
    flat: int32 array of shape (16,) — raw board values.
    Returns: float32 array of shape (16, 4, 4) — one-hot log2 encoding.

    This is the canonical state representation fed to the CNN.
    Wrap in torch.from_numpy() for a zero-copy tensor.
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
      After conv1: (B, 64,  3, 3)   ← 64 kernels of size 2×2, user-specified
      After conv2: (B, 128, 2, 2)
      After conv3: (B, 128, 1, 1)
      Flatten    : (B, 128)
      FC1        : (B, 256)
      FC2 (out)  : (B, 4)           ← Q-value per action
    """

    def __init__(self):
        super().__init__()

        # ── convolutional backbone ─────────────────────────────────────────
        self.conv1 = nn.Conv2d(N_CHANNELS, 64,  kernel_size=2)
        self.conv2 = nn.Conv2d(64,         128, kernel_size=2)
        self.conv3 = nn.Conv2d(128,        128, kernel_size=2)

        # ── fully connected head ───────────────────────────────────────────
        # After 3 × (stride-1, no-pad) 2×2 convs on a 4×4 input:
        # 4 → 3 → 2 → 1  (each conv shrinks by 1)
        # flattened dim = 128 × 1 × 1 = 128
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 4)

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
        return self.fc2(x)   # raw Q-values, no activation


# ─────────────────────────────── replay buffer ────────────────────────────────
class ReplayBuffer:
    """
    Pre-allocated circular numpy buffer.  Much faster than a Python deque of
    tensors because we do one big numpy fancy-index on every sample() call
    instead of per-transition Python object overhead.
    """

    def __init__(self, capacity: int = 100_000):
        self.cap   = capacity
        self.ptr   = 0      # next write position
        self.size  = 0      # current fill level

        self.states  = np.zeros((capacity, N_CHANNELS, 4, 4), dtype=np.float32)
        self.nstates = np.zeros((capacity, N_CHANNELS, 4, 4), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.bool_)

    def push(
        self,
        state:   np.ndarray,   # (16,4,4)
        action:  int,
        reward:  float,
        nstate:  np.ndarray,   # (16,4,4)
        done:    bool,
    ):
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
        # Convert sampled numpy slices to CPU tensors and move to DEVICE in one step
        # This avoids repeated .to(DEVICE) calls in the training loop.
        s = torch.from_numpy(self.states[idx]).to(DEVICE)
        a = torch.from_numpy(self.actions[idx]).to(DEVICE)
        r = torch.from_numpy(self.rewards[idx]).to(DEVICE)
        ns = torch.from_numpy(self.nstates[idx]).to(DEVICE)
        d = torch.from_numpy(self.dones[idx]).to(DEVICE)
        return s, a, r, ns, d

    def __len__(self) -> int:
        return self.size


# ─────────────────────────────── agent ────────────────────────────────────────
class DQNAgent:
    """
    Double DQN agent with:
      • Separate policy net  (updated every step)
      • Target net           (hard-copied every `target_sync` steps)
      • Epsilon-greedy over *valid* actions only (avoids wasting experience
        on moves that don't change the board)
      • log1p(reward) scaling  (raw scores span 0–100k+; log keeps gradients sane)
      • Huber loss + gradient clipping
    """

    def __init__(
        self,
        lr:          float = 1e-4,
        gamma:       float = 0.99,
        eps_start:   float = 1.0,
        eps_end:     float = 0.05,
        eps_decay:   float = 0.9998,   # ~log(0.05/1)/log(0.9998) ≈ 29k steps to eps_end
        batch_size:  int   = 512,
        buffer_cap:  int   = 100_000,
        target_sync: int   = 500,      # hard update every N gradient steps
        warmup:      int   = 2_000,    # steps before training begins
    ):
        self.gamma       = gamma
        self.eps         = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.batch_size  = batch_size
        self.target_sync = target_sync
        self.warmup      = warmup
        self.grad_steps  = 0

        self.policy = TwoZeroFourEightNet().to(DEVICE)
        self.target = TwoZeroFourEightNet().to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optim  = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)
        self.last_loss:    float = 0.0
        self.episode_count: int  = 0

    # ── action selection ───────────────────────────────────────────────────────
    @torch.no_grad()
    def act(self, state_np: np.ndarray, valid: list[int]) -> int:
        """
        Epsilon-greedy.  `valid` is a list of legal action indices.
        Invalid actions are masked with -inf before argmax so the network
        can't accidentally choose a wasted move.
        """
        if not valid:
            return random.randint(0, 3)

        if random.random() < self.eps:
            return random.choice(valid)

        state_t = torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)
        q = self.policy(state_t)[0]

        # Mask invalid actions
        mask = torch.full((4,), float("-inf"), device=DEVICE)
        for a in valid:
            mask[a] = q[a]
        return int(mask.argmax())

    # ── learning step ──────────────────────────────────────────────────────────
    def learn(self) -> float | None:
        if len(self.buffer) < max(self.batch_size, self.warmup):
            return None

        s, a, r, ns, done = self.buffer.sample(self.batch_size)

        # ── Double DQN ─────────────────────────────────────────────────────
        # Policy net picks the *action*, target net evaluates its *value*.
        # This decouples selection from evaluation, cutting overestimation bias.
        q_pred = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_next_a  = self.policy(ns).argmax(1, keepdim=True)
            q_next       = self.target(ns).gather(1, best_next_a).squeeze(1)
            q_target     = r + self.gamma * q_next * (~done)

        loss = F.smooth_l1_loss(q_pred, q_target)   # Huber loss

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
        self.optim.step()

        self.grad_steps += 1
        if self.grad_steps % self.target_sync == 0:
            self.target.load_state_dict(self.policy.state_dict())

        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        # detach before converting to Python scalar to avoid autograd warnings
        self.last_loss = float(loss.detach().item())
        return self.last_loss

    # ── episode runner ─────────────────────────────────────────────────────────
    def run_episode(self, train: bool = True) -> tuple[int, int, int]:
        """
        Play one complete game.

        Args:
            train: if True, push transitions to buffer and call learn().
                   Set to False for evaluation / greedy rollouts.

        Returns:
            (final_score, max_tile_value, number_of_steps)
        """
        board  = Board()
        state  = encode(board.board)
        steps  = 0
        loss   = None

        while not board.game_over:
            valid  = board.valid_actions()
            action = self.act(state, valid) if train else self._greedy(state, valid)
            dir_   = DIRS[action]

            prev_score = board.score
            board.move(dir_)
            reward     = float(board.score - prev_score)
            # +1 survival bonus keeps the agent motivated to keep the game alive
            reward    += 1.0

            next_state = encode(board.board)
            done       = board.game_over

            if train:
                self.buffer.push(
                    state, action,
                    float(np.log1p(reward)),   # log-scale: raw rewards span 0–65k+
                    next_state, done,
                )
                loss = self.learn()

            state  = next_state
            steps += 1

        max_tile = int(board.board.max())
        self.episode_count += 1
        return board.score, max_tile, steps

    @torch.no_grad()
    def _greedy(self, state_np: np.ndarray, valid: list[int]) -> int:
        """Pure greedy (no exploration) for evaluation."""
        if not valid:
            return 0
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(DEVICE)
        q = self.policy(state_t)[0]
        mask = torch.full((4,), float("-inf"), device=DEVICE)
        for a in valid:
            mask[a] = q[a]
        return int(mask.argmax())

    # ── evaluate ───────────────────────────────────────────────────────────────
    def evaluate(self, n: int = 100) -> dict:
        """Run `n` greedy games. Returns dict with score/tile stats."""
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

    # ── checkpoint ─────────────────────────────────────────────────────────────
    def save(self, path: str = "agent.pt"):
        torch.save(
            {
                "policy":        self.policy.state_dict(),
                "target":        self.target.state_dict(),
                "optim":         self.optim.state_dict(),
                "eps":           self.eps,
                "grad_steps":    self.grad_steps,
                "episode_count": self.episode_count,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.optim.load_state_dict(ckpt["optim"])
        self.eps           = ckpt["eps"]
        self.grad_steps    = ckpt.get("grad_steps", 0)
        self.episode_count = ckpt.get("episode_count", 0)


# ─────────────────────────────── training loop ────────────────────────────────
def train(
    n_episodes:  int  = 50_000,
    eval_every:  int  = 500,    # run greedy eval games
    eval_n:      int  = 50,     # how many eval games per evaluation
    save_every:  int  = 2_000,
    print_every: int  = 100,
    checkpoint:  str | None = None,
):
    """
    Main training loop.

    Typical progression with default hyperparameters:
      ~1k  episodes  → agent learns not to get stuck immediately
      ~5k  episodes  → consistently reaches 256
      ~20k episodes  → regularly reaches 512–1024
      ~50k episodes  → occasional 2048
    (Results vary; 2048 is a hard game for DQN without lookahead.)
    """
    print(f"Device: {DEVICE}")
    print(f"Network parameters: "
          f"{sum(p.numel() for p in TwoZeroFourEightNet().parameters()):,}")
    print()

    agent = DQNAgent()
    if checkpoint:
        agent.load(checkpoint)

    scores, max_tiles = [], []
    best_tile = 0

    for ep in range(1, n_episodes + 1):
        score, max_tile, steps = agent.run_episode(train=True)
        scores.append(score)
        max_tiles.append(max_tile)
        best_tile = max(best_tile, max_tile)

        # ── console log ──────────────────────────────────────────────────
        if ep % print_every == 0:
            w_scores = scores[-print_every:]
            w_tiles  = max_tiles[-print_every:]
            print(
                f"ep {ep:6,} │ "
                f"score  avg {np.mean(w_scores):7,.0f}  max {max(w_scores):7,} │ "
                f"tile  avg {np.mean(w_tiles):5.0f}  best {max(w_tiles):5} │ "
                f"ε {agent.eps:.4f} │ "
                f"buf {len(agent.buffer):,}"
            )

        # ── greedy evaluation ─────────────────────────────────────────────
        if ep % eval_every == 0:
            stats = agent.evaluate(eval_n)
            print(
                f"\n  ── eval ({eval_n} greedy games) ──\n"
                f"  mean score {stats['mean_score']:,.0f} │ "
                f"max score {stats['max_score']:,} │ "
                f"max tile {stats['max_tile']} │ "
                f"tile dist: {stats['tile_counts']}\n"
            )

        # ── checkpoint ────────────────────────────────────────────────────
        if ep % save_every == 0:
            agent.save(f"agent_ep{ep:06d}.pt")

    agent.save("agent_final.pt")
    return agent


# ─────────────────────────────── entry point ──────────────────────────────────
if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    train(checkpoint=ckpt)
