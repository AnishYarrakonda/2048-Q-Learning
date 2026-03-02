import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Optional, Tuple, List

from board import Board, device

# ---------------------------------------------------------------------------
# Neural Network — Dueling DQN with same-padding convolutions
# ---------------------------------------------------------------------------

class ConnectFourNet(nn.Module):
    def __init__(self):
        super().__init__()

        # --- convolutional backbone (same padding → preserves 6x7 throughout) ---
        # 3x3 kernel with padding=1 is the standard "same" conv for odd kernels.
        self.conv1 = nn.Conv2d(2,   128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # Flattened: 128 channels * 6 rows * 7 cols = 5376
        self._flat_size = 128 * Board.ROWS * Board.COLS  # 5376

        # --- shared trunk ---
        self.fc_shared1 = nn.Linear(self._flat_size, 512)
        self.fc_shared2 = nn.Linear(512, 512)

        # --- value stream: board state → scalar ---
        self.value_fc1 = nn.Linear(512, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # --- advantage stream: board state → 7 move advantages ---
        self.adv_fc1   = nn.Linear(512, 256)
        self.adv_fc2   = nn.Linear(256, Board.COLS)

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
        x: (batch, 2, 6, 7) — already on the correct device
        returns: (batch, 7) Q-values
        """
        # Conv backbone — spatial size stays 6x7 throughout
        x = F.relu(self.bn1(self.conv1(x)))   # → (b, 128, 6, 7)
        x = F.relu(self.bn2(self.conv2(x)))   # → (b, 128, 6, 7)
        x = F.relu(self.bn3(self.conv3(x)))   # → (b, 128, 6, 7)

        x = x.flatten(start_dim=1)            # → (b, 5376)

        # Shared trunk
        x = F.relu(self.fc_shared1(x))        # → (b, 512)
        x = F.relu(self.fc_shared2(x))        # → (b, 512)

        # Dueling heads
        v = F.relu(self.value_fc1(x))         # → (b, 256)
        v = self.value_fc2(v)                  # → (b, 1)

        a = F.relu(self.adv_fc1(x))           # → (b, 256)
        a = self.adv_fc2(a)                    # → (b, 7)

        # Combine: Q = V + A - mean(A)
        # Subtracting mean(A) makes the decomposition identifiable —
        # otherwise V and A can both drift without changing Q.
        q = v + a - a.mean(dim=1, keepdim=True)  # → (b, 7)
        return q

    def policy(self, board: Board) -> torch.Tensor:
        """
        Returns a masked softmax probability distribution over valid columns.
        Returns a (7,) tensor on `device`.
        """
        self.eval()
        with torch.no_grad():
            state = Board.board_to_tensor(board).view(1, 2, Board.ROWS, Board.COLS)
            logits = self(state).squeeze(0)  # (7,)

            mask = torch.full((Board.COLS,), float('-inf'), device=device)
            for col in board.valid_moves():
                mask[col] = 0.0

            probs = F.softmax(logits + mask, dim=0)
        return probs

    def best_move(self, board: Board) -> int:
        """Return the column with the highest Q-value among valid moves."""
        self.eval()
        with torch.no_grad():
            state = Board.board_to_tensor(board).view(1, 2, Board.ROWS, Board.COLS)
            q = self(state).squeeze(0)  # (7,)

            valid = board.valid_moves()
            mask = torch.full((Board.COLS,), float('-inf'), device=device)
            for col in valid:
                mask[col] = 0.0

            return int((q + mask).argmax().item())

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

class SumTree:
    """SumTree for prioritized sampling. Stores priorities in a binary tree."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float32)
        self.data = [None] * self.capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, p: float, data_item) -> int:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data_item
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
        return idx

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Sum-tree based Prioritized Experience Replay (PER).

    Stores tuples (state, action, reward, next_state, done) and samples by priority.
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.eps = eps
        self.tree = SumTree(self.capacity)

    def __len__(self) -> int:
        return int(self.tree.n_entries)

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        # New priorities get max priority so they are likely to be sampled
        max_p = np.max(self.tree.tree[-self.tree.capacity:]) if self.tree.n_entries > 0 else 1.0
        if max_p == 0:
            max_p = 1.0
        data = (state.detach().cpu(), action, float(reward), next_state.detach().cpu(), bool(done))
        self.tree.add(max_p, data) # type: ignore

    def sample(self, batch_size: int, beta: float = 0.4):
        N = self.tree.n_entries
        if N == 0:
            raise ValueError("Trying to sample from empty buffer")

        batch = []
        idxs = np.empty(batch_size, dtype=np.int32)
        ps = np.empty(batch_size, dtype=np.float32)

        total = self.tree.total()
        # If total is zero (unexpected), fallback to uniform sampling over filled entries
        if total <= 0.0:
            choices = np.random.randint(0, N, size=batch_size)
            for i, di in enumerate(choices):
                tree_idx = int(di + self.tree.capacity - 1)
                p = float(self.tree.tree[tree_idx])
                data = self.tree.data[di]
                idxs[i] = tree_idx
                ps[i] = p
                batch.append(data)
        else:
            segment = total / batch_size
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)
                idx, p, data = self.tree.get(s)
                # If data is None for some reason, fallback to a random filled slot
                if data is None:
                    di = random.randint(0, N - 1)
                    idx = int(di + self.tree.capacity - 1)
                    p = float(self.tree.tree[idx])
                    data = self.tree.data[di]
                idxs[i] = int(idx)
                ps[i] = float(p)
                batch.append(data)

        # normalize probabilities and compute importance-sampling weights
        probs = ps / (self.tree.total() + 1e-12)
        probs = np.clip(probs, 1e-12, None)
        weights = (N * probs) ** (-beta)
        weights = weights / (weights.max() + 1e-12)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.cat(states).to(device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.cat(next_states).to(device),
            torch.tensor(dones, dtype=torch.float32, device=device),
            idxs,
            torch.tensor(weights, dtype=torch.float32, device=device),
        )

    def update_priorities(self, idxs: List[int], td_errors: np.ndarray) -> None:
        # td_errors: array-like of size len(idxs)
        for idx, err in zip(idxs, td_errors):
            p = (abs(float(err)) + self.eps) ** self.alpha
            self.tree.update(int(idx), float(p))


# ---------------------------------------------------------------------------
# DQN Agent — Double DQN + Dueling network
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 128,
        target_update_freq: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10_000,
        buffer_capacity: int = 100_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
    ):
        self.gamma               = gamma
        self.batch_size          = batch_size
        self.target_update_freq  = target_update_freq
        self.epsilon_start       = epsilon_start
        self.epsilon_end         = epsilon_end
        self.epsilon_decay       = epsilon_decay
        self.steps_done          = 0
        # PER parameters
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames

        # Epsilon bump (promotion reset) tracking
        self._epsilon_bump = None  # dict with keys start_value, decay_steps, start_step

        self.policy_net = ConnectFourNet().to(device)
        self.target_net = ConnectFourNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)
        self.buffer    = PrioritizedReplayBuffer(buffer_capacity, alpha=self.per_alpha)

    @property
    def epsilon(self) -> float:
        # If an epsilon bump is active, use that schedule for the specified decay window.
        if self._epsilon_bump is not None:
            start_value = self._epsilon_bump["start_value"]
            decay_steps = self._epsilon_bump["decay_steps"]
            start_step = self._epsilon_bump["start_step"]
            elapsed = max(0, self.steps_done - start_step)
            if elapsed <= decay_steps:
                e = self.epsilon_end + (start_value - self.epsilon_end) * np.exp(-elapsed / decay_steps)
                return float(e)
            else:
                # bump finished
                self._epsilon_bump = None

        e = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.steps_done / self.epsilon_decay)
        return float(e)

    def select_action(self, board: Board) -> int:
        valid = board.valid_moves()
        if random.random() < self.epsilon:
            return random.choice(valid)
        return self.policy_net.best_move(board)

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        # Beta annealing
        beta = self.per_beta_start + (1.0 - self.per_beta_start) * min(1.0, self.steps_done / max(1, self.per_beta_frames))

        states, actions, rewards, next_states, dones, idxs, is_weights = self.buffer.sample(self.batch_size, beta=beta)

        s  = states.view(-1, 2, Board.ROWS, Board.COLS)
        ns = next_states.view(-1, 2, Board.ROWS, Board.COLS)

        self.policy_net.train()
        q_values = self.policy_net(s).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: policy net selects, target net evaluates
            next_actions = self.policy_net(ns).argmax(dim=1, keepdim=True)
            next_q = self.target_net(ns).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        # element-wise loss
        elementwise_loss = F.smooth_l1_loss(q_values, target, reduction='none')
        loss = (is_weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # update priorities
        with torch.no_grad():
            td_errors = (target - q_values).abs().detach().cpu().numpy()
        # idxs are tree indices; update with td_errors
        self.buffer.update_priorities(idxs, td_errors) # type: ignore

        self.steps_done += 1

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

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

    def start_epsilon_bump(self, start_value: float = 0.25, decay_steps: int = 2000) -> None:
        """Temporarily bump epsilon to `start_value` and decay it back to `epsilon_end` over `decay_steps` learn steps."""
        self._epsilon_bump = {
            "start_value": float(start_value),
            "decay_steps": int(decay_steps),
            "start_step": int(self.steps_done),
        }


if __name__ == "__main__":
    board = Board()
    agent = DQNAgent()

    print(f"Device: {device}")
    total_params = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"Policy net params: {total_params:,}")

    dummy = Board.board_to_tensor(board).view(1, 2, Board.ROWS, Board.COLS)
    out   = agent.policy_net(dummy)
    print(f"Output shape: {out.shape}")   # torch.Size([1, 7])

    action = agent.select_action(board)
    print(f"Selected action: {action}")
    print("Sanity check passed ✓")