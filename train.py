# imports
import time
import torch
from agent import Agent
from board import Board
from typing import TypedDict


class Config(TypedDict):
    layers: list[int]
    lr: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    gamma: float
    num_episodes: int
    train: bool
    watch_game: bool
    watch_steps: int
    watch_delay: float
    progress_interval: int
    save_enabled: bool
    save_interval: int
    save_prefix: str


DEFAULT_CONFIG: Config = {
    "layers": [128, 64],
    "lr": 0.001,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "gamma": 0.95,
    "num_episodes": 1000,
    "train": True,
    "watch_game": False,
    "watch_steps": 1,
    "watch_delay": 0.2,
    "progress_interval": 25,
    "save_enabled": True,
    "save_interval": 500,
    "save_prefix": "agent_checkpoint",
}


def ask_yes_no(prompt: str, default: bool) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_hint}]: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y/yes or n/no.")


def ask_int(prompt: str, default: int, minimum: int | None = None) -> int:
    while True:
        raw = input(f"{prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
            if minimum is not None and value < minimum:
                print(f"Please enter an integer >= {minimum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer.")


def ask_float(prompt: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    while True:
        raw = input(f"{prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            value = float(raw)
            if minimum is not None and value < minimum:
                print(f"Please enter a value >= {minimum}.")
                continue
            if maximum is not None and value > maximum:
                print(f"Please enter a value <= {maximum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def ask_layers(default: list[int]) -> list[int]:
    while True:
        raw = input(
            f"Hidden layer sizes (comma-separated, e.g. 128,64) [default={','.join(map(str, default))}]: "
        ).strip()
        if raw == "":
            return default
        try:
            values = [int(piece.strip()) for piece in raw.split(",") if piece.strip() != ""]
            if not values:
                print("Please provide at least one hidden layer size.")
                continue
            if any(v <= 0 for v in values):
                print("Layer sizes must be positive integers.")
                continue
            return values
        except ValueError:
            print("Please enter comma-separated integers like: 256,128,64")


def build_runtime_config() -> Config:
    print("Connect Four Training Configuration")
    print("Press Enter on any prompt to use its default value.")
    print()

    use_defaults = ask_yes_no("Use all default settings?", True)
    if use_defaults:
        return DEFAULT_CONFIG.copy()

    config: Config = DEFAULT_CONFIG.copy()

    print("\nAgent settings")
    config["layers"] = ask_layers(config["layers"])
    config["lr"] = ask_float("Learning rate", float(config["lr"]), minimum=0.0)
    config["epsilon"] = ask_float("Initial epsilon", float(config["epsilon"]), minimum=0.0, maximum=1.0)
    config["epsilon_decay"] = ask_float("Epsilon decay per update", float(config["epsilon_decay"]), minimum=0.0, maximum=1.0)
    config["epsilon_min"] = ask_float("Minimum epsilon", float(config["epsilon_min"]), minimum=0.0, maximum=1.0)
    config["gamma"] = ask_float("Discount factor gamma", float(config["gamma"]), minimum=0.0, maximum=1.0)

    print("\nTraining settings")
    config["train"] = ask_yes_no("Enable training updates?", bool(config["train"]))
    config["num_episodes"] = ask_int("Number of episodes", int(config["num_episodes"]), minimum=1)
    config["progress_interval"] = ask_int("Print training progress every N episodes", int(config["progress_interval"]), minimum=1)

    print("\nVisualization settings")
    config["watch_game"] = ask_yes_no("Watch board states while running?", bool(config["watch_game"]))
    if bool(config["watch_game"]):
        config["watch_steps"] = ask_int("Show board every N turns", int(config["watch_steps"]), minimum=1)
        config["watch_delay"] = ask_float("Delay between printed boards (seconds)", float(config["watch_delay"]), minimum=0.0)

    print("\nCheckpoint settings")
    config["save_enabled"] = ask_yes_no("Save model checkpoints during training?", bool(config["save_enabled"]))
    if bool(config["save_enabled"]):
        config["save_interval"] = ask_int("Save checkpoint every N episodes", int(config["save_interval"]), minimum=1)
        prefix_default = str(config["save_prefix"])
        raw_prefix = input(f"Checkpoint filename prefix [default={prefix_default}]: ").strip()
        if raw_prefix != "":
            config["save_prefix"] = raw_prefix

    print()
    return config


def play_game(
    agent: Agent,
    train: bool = True,
    watch_game: bool = False,
    watch_steps: int = 1,
    watch_delay: float = 0.2,
) -> int:
    board = Board()
    done = False
    winner = 0

    while not done:
        valid_moves = board.valid_moves()
        if not valid_moves:
            winner = 0
            break

        acting_player = 1 if board.turn % 2 == 0 else 2
        state = Board.board_to_tensor(board=board)
        action = agent.select_action(board=board, valid_moves=valid_moves)

        row = board.make_move(action)
        if row is None:
            done = True
            winner = 2 if acting_player == 1 else 1
            if train:
                next_state = Board.board_to_tensor(board=board)
                agent.train_step(state, action, -1.0, next_state, done)
            break

        done, winner = board.game_over(row, action)

        if watch_game and watch_steps > 0 and (board.turn % watch_steps == 0):
            print(board)
            if watch_delay > 0:
                time.sleep(watch_delay)

        if train:
            reward = 0.0
            if done and winner == acting_player:
                reward = 1.0
            elif done and winner != 0:
                reward = -1.0
            next_state = Board.board_to_tensor(board=board)
            agent.train_step(state, action, reward, next_state, done)

    return winner


def run_training(config: Config) -> None:
    agent = Agent(
        layers=list(config["layers"]),
        lr=float(config["lr"]),
        epsilon=float(config["epsilon"]),
        epsilon_decay=float(config["epsilon_decay"]),
        epsilon_min=float(config["epsilon_min"]),
        gamma=float(config["gamma"]),
    )

    num_episodes = int(config["num_episodes"])
    should_train = bool(config["train"])
    progress_interval = int(config["progress_interval"])
    watch_game = bool(config["watch_game"])
    watch_steps = int(config["watch_steps"])
    watch_delay = float(config["watch_delay"])
    save_enabled = bool(config["save_enabled"])
    save_interval = int(config["save_interval"])
    save_prefix = str(config["save_prefix"])

    stats = {"wins": 0, "losses": 0, "draws": 0}

    for episode in range(num_episodes):
        winner = play_game(
            agent=agent,
            train=should_train,
            watch_game=watch_game,
            watch_steps=watch_steps,
            watch_delay=watch_delay,
        )

        if winner == 1:
            stats["wins"] += 1
        elif winner == 2:
            stats["losses"] += 1
        else:
            stats["draws"] += 1

        if (episode + 1) % progress_interval == 0:
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"wins={stats['wins']} losses={stats['losses']} draws={stats['draws']} | "
                f"epsilon={agent.epsilon:.4f}"
            )

        if save_enabled and (episode + 1) % save_interval == 0:
            path = f"{save_prefix}_{episode + 1}.pt"
            torch.save(agent.model.state_dict(), path)
            print(f"Saved checkpoint: {path}")

    print("\nTraining complete.")
    print(f"Final stats after {num_episodes} episodes: {stats}")

    save_final = ask_yes_no("Save final model?", True)
    if save_final:
        final_name = input("Final model filename [default=agent_final.pt]: ").strip() or "agent_final.pt"
        torch.save(agent.model.state_dict(), final_name)
        print(f"Saved final model: {final_name}")


if __name__ == "__main__":
    runtime_config = build_runtime_config()
    run_training(runtime_config)
