# imports
import random
from collections import deque
from typing import Optional

import torch
import torch.nn as nn

from board import Board


class ConnectFourCNN(nn.Module):
    """CNN Q-network consuming flattened, perspective-aligned 2*6*7 input."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        conv_out = 128 * Board.ROWS * Board.COLS
        self.head = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, Board.COLS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.view(-1, 2, Board.ROWS, Board.COLS)
        features = self.conv(spatial).flatten(1)
        return self.head(features)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple[torch.Tensor, int, float, torch.Tensor, bool]]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class Agent:
    BATCH_SIZE = 256
    MIN_BUFFER = 1000
    TARGET_UPDATE_FREQ = 500

    def __init__(
        self: "Agent",
        lr: float = 0.001,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.997,
        epsilon_min: float = 0.05,
        gamma: float = 0.97,
    ) -> None:
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model: nn.Module = ConnectFourCNN().to(self.device)
        self.target_model: nn.Module = ConnectFourCNN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(capacity=50_000)
        self._update_counter = 0
        self._push_count = 0
        self.model.eval()

    def predict(self: "Agent", board: Board) -> torch.Tensor:
        state = Board.board_to_tensor(board).to(self.device)
        with torch.no_grad():
            return self.model(state)

    def select_action(self: "Agent", board: Board, valid_moves: Optional[list[int]] = None) -> int:
        if valid_moves is None:
            valid_moves = board.valid_moves()
        if not valid_moves:
            return 0
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        state = Board.board_to_tensor(board).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)[0]
        masked = q_values.clone()
        for col in range(Board.COLS):
            if col not in valid_moves:
                masked[col] = -float("inf")
        return int(torch.argmax(masked).item())

    def train_step(
        self: "Agent",
    ) -> None:
        if len(self.replay_buffer) < self.MIN_BUFFER:
            return

        transitions = self.replay_buffer.sample(self.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*transitions)

        state_batch = torch.stack(list(states)).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_batch = torch.stack(list(next_states)).to(self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)

        self.model.train()
        q_values = self.model(state_batch)
        q_selected = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_state_batch)
            max_next_q = next_q_values.max(dim=1).values
            target_values = reward_batch + (1.0 - done_batch) * self.gamma * max_next_q

        loss = self.loss_fn(q_selected, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._update_counter += 1
        if self._update_counter % self.TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()

    def push(
        self: "Agent",
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        cpu_state = state.detach().cpu()
        cpu_next_state = next_state.detach().cpu()
        if cpu_state.dim() > 1:
            cpu_state = cpu_state.squeeze(0)
        if cpu_next_state.dim() > 1:
            cpu_next_state = cpu_next_state.squeeze(0)
        self.replay_buffer.push(cpu_state, action, float(reward), cpu_next_state, bool(done))
        self._push_count += 1
