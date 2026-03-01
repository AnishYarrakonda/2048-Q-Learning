# imports
import random
from typing import Union
from board import Board
import torch
import torch.nn as nn
import torch.nn.functional as F

# use gpu for faster operations
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# agent object
class Agent:
    
    # initialize the agent
    def __init__(self, layers: list[int], lr=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.95):
        # stores all of the layers to be unpacked into the model
        modules: list[nn.Module] = [nn.Flatten()]

        input_size = 2 * 6 * 7  # input dimension

        # add hidden layers dynamically
        for hidden_size in layers:
            modules.append(nn.Linear(input_size, hidden_size))
            modules.append(nn.ReLU())
            input_size = hidden_size

        # output layer
        modules.append(nn.Linear(input_size, 7))  # 7 possible moves

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
    def select_action(self: "Agent", board: Board) -> int:
        # exploration
        if random.random() < self.epsilon:
            return random.randint(0, 6)

        # exploitation
        state = Board.board_to_tensor(board)
        q_values = self.model(state).detach().flatten()
        return torch.argmax(q_values).item() # type: ignore


    # train on a single step using TD update
    def train_step(
        self: "Agent",
        state: torch.Tensor,
        action: int,
        reward: Union[float, torch.Tensor],
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        # 1. predict Q-values for current state
        q_values = self.model(state)

        # 2. predict Q-values for next state
        with torch.no_grad():
            next_q_values = self.model(next_state)

        # 3. compute target for chosen action
        if done:
            target = reward
        else:
            target = reward + self.gamma * torch.max(next_q_values)

        # 4. compute loss only for chosen action using self.loss_fn
        loss = self.loss_fn(q_values[0, action], target)

        # 5. backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6. decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
