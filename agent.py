import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Optional

from board import Board, device

# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------
# Input:  1 x 2 x 6 x 7  (batch x channels x rows x cols)
#         channel 0 = current player's pieces
#         channel 1 = opponent's pieces
#
# Architecture:
#   3x Conv layers (128 filters, 4x4 kernel, padding=1) with BatchNorm + ReLU
#   Flatten → FC 512 → FC 256 → FC 7  (one logit per column)
#
# After each conv the spatial size shrinks:
#   6x7  →  5x6  →  4x5  →  3x4   (128 channels throughout)
#   Flattened: 128 * 3 * 4 = 1536
# ---------------------------------------------------------------------------

class ConnectFourNet(nn.Module):
    def __init__(self):
        super().__init__()

        # --- convolutional backbone ---
        self.conv1 = nn.Conv2d(2,   128, kernel_size=4, padding=1)
        self.bn1   = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=4, padding=1)
        self.bn2   = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # --- fully-connected head ---
        # 128 channels * 3 rows * 4 cols = 1536 after 3 conv layers
        self._flat_size = 128 * 3 * 4

        self.fc1 = nn.Linear(self._flat_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, Board.COLS)   # 7 output logits

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 2, 6, 7)  — already on the correct device
        returns: (batch, 7) raw logits
        """
        x = F.relu(self.bn1(self.conv1(x)))   # → (b, 128, 5, 6)
        x = F.relu(self.bn2(self.conv2(x)))   # → (b, 128, 4, 5)
        x = F.relu(self.bn3(self.conv3(x)))   # → (b, 128, 3, 4)

        x = x.flatten(start_dim=1)            # → (b, 1536)

        x = F.relu(self.fc1(x))               # → (b, 512)
        x = F.relu(self.fc2(x))               # → (b, 256)
        x = self.fc3(x)                       # → (b, 7)
        return x

    def policy(self, board: Board) -> torch.Tensor:
        """
        Convenience: given a Board, return a masked softmax probability
        distribution over valid columns only (invalid cols → 0).
        Returns a (7,) tensor on `device`.
        """
        self.eval()
        with torch.no_grad():
            # board_to_tensor already reshapes to (1, 2*6*7) — reshape to (1,2,6,7)
            state = Board.board_to_tensor(board).view(1, 2, Board.ROWS, Board.COLS)
            logits = self(state).squeeze(0)  # (7,)

            # mask illegal moves
            mask = torch.full((Board.COLS,), float('-inf'), device=device)
            for col in board.valid_moves():
                mask[col] = 0.0

            probs = F.softmax(logits + mask, dim=0)
        return probs

    def best_move(self, board: Board) -> int:
        """Return the column with the highest probability."""
        probs = self.policy(board)
        return int(probs.argmax().item())

    def sample_move(self, board: Board, temperature: float = 1.0) -> int:
        """Sample a move proportional to policy probabilities."""
        probs = self.policy(board)
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1).item())


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Stores (state_tensor, action, reward, next_state_tensor, done) tuples."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.cat(states).to(device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.cat(next_states).to(device),
            torch.tensor(dones,   dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent  (wraps the network + replay buffer + optimizer)
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        target_update_freq: int = 500,   # steps between hard target-net updates
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,     # steps over which epsilon anneals
        buffer_capacity: int = 50_000,
    ):
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.steps_done = 0

        self.policy_net = ConnectFourNet().to(device)
        self.target_net = ConnectFourNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

    # ---- epsilon schedule ------------------------------------------------

    @property
    def epsilon(self) -> float:
        decay = self.epsilon_decay
        e = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.steps_done / decay)
        return float(e)

    # ---- action selection ------------------------------------------------

    def select_action(self, board: Board) -> int:
        """Epsilon-greedy action selection."""
        valid = board.valid_moves()
        if random.random() < self.epsilon:
            return random.choice(valid)
        return self.policy_net.best_move(board)

    # ---- learning step ---------------------------------------------------

    def learn(self) -> Optional[float]:
        """Sample a mini-batch and perform one gradient step. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Reshape flat tensors → (b, 2, 6, 7)
        s  = states.view(-1, 2, Board.ROWS, Board.COLS)
        ns = next_states.view(-1, 2, Board.ROWS, Board.COLS)

        # Q(s, a)
        self.policy_net.train()
        q_values = self.policy_net(s).gather(1, actions.unsqueeze(1)).squeeze(1)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1

        # Hard update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ---- persistence -----------------------------------------------------

    def save(self, path: str = "agent.pth"):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)
        print(f"[Agent] Saved → {path}")

    def load(self, path: str = "agent.pth"):
        ckpt = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt.get("steps_done", 0)
        print(f"[Agent] Loaded ← {path}  (step {self.steps_done})")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    board = Board()
    agent = DQNAgent()

    print(f"Device: {device}")
    print(f"Policy net params: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

    # forward-pass smoke test
    dummy = Board.board_to_tensor(board).view(1, 2, Board.ROWS, Board.COLS)
    out   = agent.policy_net(dummy)
    print(f"Output shape: {out.shape}")   # should be torch.Size([1, 7])

    # action selection
    action = agent.select_action(board)
    print(f"Selected action: {action}")

    print("Sanity check passed ✓")