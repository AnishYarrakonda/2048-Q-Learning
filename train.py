# imports
import torch
import time
from board import Board
from agent import Agent

# simulate a single game
def play_game(agent: Agent, train=True):
    board = Board()     # make the board
    done = False        # track if game is done
    winner = 0          # store the winner

    while not done:
        # get the state of the game for the agent to make actions
        state = Board.board_to_tensor(board=board)

        # choose an action
        action = agent.select_action(board=board)

        # make the move
        board.make_move(action)

        # check if game is over
        done, winner = board.is_game_over()

        if train:
            # get reward (1 for win, -1 for loss, 0 for draw/neither)
            reward = 0
            if winner == 1:
                reward = 1
            elif winner == -1:
                reward = -1

            next_state = Board.board_to_tensor(board=board)
            agent.train_step(state, action, reward, next_state, done)




if __name__ == "__main__":
    # create a new agent to train
    agent_1 : Agent = Agent(layers=[128, 64], lr=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.95)