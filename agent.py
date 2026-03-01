# imports
import random
from board import Board
import torch
import torch.nn as nn
import torch.nn.functional as F

# use gpu for faster operations
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# agent object
class Agent:
    
    # initialize the agent
    def __init__(self):
        # neural network model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2*6*7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=7)
        )

        # optimizer for training the model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # exploration rate for epsilon-greedy action selection
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # gamma for discounting future rewards
        self.gamma = 0.95


    # predict the best move given the current board state
    def predict(self, board: Board):
        state = Board.board_to_tensor(board)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values
    

    # epsilon-greedy action selection
    def select_action(self, board: Board):
        # exploration
        if random.random() < self.epsilon:
            return random.randint(0, 6)

        # exploitation
        state = Board.board_to_tensor(board)
        q_values = self.model(state).detach().flatten()
        return torch.argmax(q_values).item()


    # train on a single step using TD update
    def train_step(self, state, action, reward, next_state, done):
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

        # 4. compute loss only for chosen action
        loss = F.mse_loss(q_values[0, action], target)

        # 5. backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6. decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
