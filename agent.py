# imports
import random
from typing import Optional
from board import Board
import torch
import torch.nn as nn
import torch.nn.functional as F

# agent object
class Agent:
    
    # initialize the agent
    def __init__(
        self: "Agent",
        layers: list[int],
        lr: float = 0.001,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        gamma: float = 0.95,
    ) -> None:
        # use gpu for faster operations
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # stores all of the layers to be unpacked into the model
        modules: list[nn.Module] = []

        input_size = 2 * 6 * 7 + 1  # input dimension (flattened board + scalar turn value)

        # add hidden layers dynamically
        for hidden_size in layers:
            modules.append(nn.Linear(input_size, hidden_size, device=self.device))
            modules.append(nn.ReLU())
            input_size = hidden_size

        # output layer
        modules.append(nn.Linear(input_size, 7, device=self.device))  # 7 possible moves

        # combine all modules
        self.model = nn.Sequential(*modules)

        # optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # epsilon-greedy parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # discount factor
        self.gamma = gamma


    # predict the best move given the current board state
    def predict(self: "Agent", board: Board) -> torch.Tensor:
        state = Board.board_to_tensor(board)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values
    

    # epsilon-greedy action selection
    def select_action(self: "Agent", board: Board, valid_moves: Optional[list[int]] = None) -> int:
        if valid_moves is None:
            valid_moves = board.valid_moves()

        if not valid_moves:
            return 0  # fallback safety (game should already be done)

        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        state = Board.board_to_tensor(board).to(self.device)
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state)[0]

        # mask invalid moves
        masked_q = q_values.clone()
        for col in range(7):
            if col not in valid_moves:
                masked_q[col] = -float("inf")

        return torch.argmax(masked_q).item() # type: ignore

    # train on a single step using TD update
    def train_step(
            self: "Agent",
            state: torch.Tensor,
            action: int,
            reward: float,
            next_state: torch.Tensor,
            done: bool,
        ) -> None:

            # move tensors to correct device
            state = state.to(self.device)
            next_state = next_state.to(self.device)

            # ensure batch dimension (model expects [batch, features])
            if state.dim() == 3:
                state = state.unsqueeze(0)
            if next_state.dim() == 3:
                next_state = next_state.unsqueeze(0)

            # 1. predict Q-values for current state
            q_values = self.model(state)

            # 2. compute target Q-values
            with torch.no_grad():
                next_q_values = self.model(next_state)

            # clone current predictions so we only modify chosen action
            target_q = q_values.clone().detach()

            if done:
                target_value = reward
            else:
                max_next_q = torch.max(next_q_values)
                target_value = reward + self.gamma * max_next_q.item()

            # convert to tensor on correct device
            target_q[0, action] = torch.tensor(
                target_value, dtype=torch.float32, device=self.device
            )

            # 3. compute loss across full Q vector
            loss = self.loss_fn(q_values, target_q)

            # 4. backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 5. decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
