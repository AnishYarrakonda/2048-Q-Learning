# imports
import torch
import time
from board import Board
from agent import Agent

# simulate a single game
def play_game(agent: Agent, train=True):
    board = Board()
    done = False
    winner = 0

    while not done:
        state = Board.board_to_tensor(board=board)

        action = agent.select_action(board=board)