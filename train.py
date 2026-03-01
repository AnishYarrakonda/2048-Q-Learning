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
        row = board.make_move(action)

        # check if game is over (win/draw)
        done, winner = board.game_over(row, action)

        # always print the board
        print(board)
        time.sleep(0.1)  # small delay for visualization

        if train:
            # get reward (1 for win, -1 for loss, 0 for draw/neither)
            reward = 0
            if winner == 1:
                reward = 1
            elif winner == -1:
                reward = -1

            # get the next state for training
            next_state = Board.board_to_tensor(board=board)

            # train the agent for this move
            agent.train_step(state, action, reward, next_state, done)

    return winner


# train a connect four model
if __name__ == "__main__":
    # create a new agent to train
    agent_1: Agent = Agent(
        layers=[128, 64],           # hidden layers
        lr=0.001,                   # learning rate
        epsilon=1.0,                # initial exploration rate
        epsilon_decay=0.995,        # epsilon decay
        epsilon_min=0.01,           # minimum epsilon
        gamma=0.95                  # discount factor
    )

    # training parameters
    num_episodes = 1000           # number of games to train
    stats = {'wins':0, 'losses':0, 'draws':0}

    # training loop
    for episode in range(num_episodes):
        winner = play_game(agent_1, train=True)

        # update stats
        if winner == 1:
            stats['wins'] += 1
        elif winner == -1:
            stats['losses'] += 1
        else:
            stats['draws'] += 1

        # decay epsilon after each episode
        if agent_1.epsilon > agent_1.epsilon_min:
            agent_1.epsilon *= agent_1.epsilon_decay

        # print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1} completed: {stats}")

        # optional: save model every 500 episodes
        if (episode + 1) % 500 == 0:
            torch.save(agent_1.model.state_dict(), f"agent_checkpoint_{episode+1}.pt")
            print(f"Saved model checkpoint at episode {episode + 1}")

    print("Training complete!")
    print(f"Final stats after {num_episodes} games: {stats}")