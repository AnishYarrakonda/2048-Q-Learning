# imports
import os
import time
from typing import TypedDict

import torch

from agent import Agent
from board import Board


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
    run_name: str
    save_dir: str
    save_final: bool
    final_model_path: str


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
    "watch_steps": 100,
    "watch_delay": 0.2,
    "progress_interval": 25,
    "save_enabled": True,
    "save_interval": 500,
    "run_name": "agent",
    "save_dir": "models",
    "save_final": True,
    "final_model_path": "models/agent_final.pt",
}


class ANSI:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"


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


def ask_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [default={default}]: ").strip()
    return raw if raw else default


def ask_layers(default: list[int]) -> list[int]:
    default_label = ",".join(map(str, default))
    while True:
        raw = input(
            f"Hidden layer sizes (comma-separated, e.g. 128,64) [default={default_label}]: "
        ).strip()
        if raw == "":
            return default
        try:
            values = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
            if not values:
                print("Please provide at least one hidden layer size.")
                continue
            if any(v <= 0 for v in values):
                print("Layer sizes must be positive integers.")
                continue
            return values
        except ValueError:
            print("Please enter comma-separated integers like: 256,128,64")


def section_default(section_name: str) -> bool:
    raw = input(
        f"\n{section_name}: type 'default' to keep this section default, or press Enter to customize: "
    ).strip().lower()
    return raw == "default"


def configure_agent(config: Config) -> None:
    if section_default("Agent Settings"):
        print("Using default agent settings.")
        return
    config["layers"] = ask_layers(config["layers"])
    config["lr"] = ask_float("Learning rate", config["lr"], minimum=0.0)
    config["epsilon"] = ask_float("Initial epsilon", config["epsilon"], minimum=0.0, maximum=1.0)
    config["epsilon_decay"] = ask_float(
        "Epsilon decay per update", config["epsilon_decay"], minimum=0.0, maximum=1.0
    )
    config["epsilon_min"] = ask_float("Minimum epsilon", config["epsilon_min"], minimum=0.0, maximum=1.0)
    config["gamma"] = ask_float("Discount factor gamma", config["gamma"], minimum=0.0, maximum=1.0)


def configure_training(config: Config) -> None:
    if section_default("Training Settings"):
        print("Using default training settings.")
        return
    config["train"] = ask_yes_no("Enable training updates?", config["train"])
    config["num_episodes"] = ask_int("Number of episodes", config["num_episodes"], minimum=1)
    config["progress_interval"] = ask_int(
        "Print progress every N episodes", config["progress_interval"], minimum=1
    )


def configure_visuals(config: Config) -> None:
    if section_default("Visualization Settings"):
        print("Using default visualization settings.")
        return
    config["watch_game"] = ask_yes_no("Watch board states while running?", config["watch_game"])
    if config["watch_game"]:
        config["watch_steps"] = ask_int("Show board every N turns", config["watch_steps"], minimum=1)
        config["watch_delay"] = ask_float(
            "Delay between board prints (seconds)", config["watch_delay"], minimum=0.0
        )


def configure_saving(config: Config) -> None:
    if section_default("Saving Settings"):
        print("Using default saving settings.")
        return
    config["save_enabled"] = ask_yes_no("Save checkpoints during training?", config["save_enabled"])
    config["save_interval"] = ask_int("Save checkpoint every N episodes", config["save_interval"], minimum=1)
    config["run_name"] = ask_text("Run/model name (used in checkpoint filenames)", config["run_name"])
    config["save_dir"] = ask_text("Directory for checkpoints and model files", config["save_dir"])
    config["save_final"] = ask_yes_no("Save final model when training ends?", config["save_final"])
    default_final = f"{config['save_dir']}/{config['run_name']}_final.pt"
    config["final_model_path"] = ask_text("Final model path", default_final)


def build_runtime_config() -> Config:
    print("Connect Four Training Configuration")
    print("Press Enter on any prompt to accept that prompt's default.")
    print("For each section, type 'default' to skip prompts and keep the whole section default.")

    config: Config = DEFAULT_CONFIG.copy()
    configure_agent(config)
    configure_training(config)
    configure_visuals(config)
    configure_saving(config)

    # Keep final model path aligned when untouched.
    if config["final_model_path"] == DEFAULT_CONFIG["final_model_path"]:
        config["final_model_path"] = f"{config['save_dir']}/{config['run_name']}_final.pt"

    return config


def format_progress_line(
    episode: int,
    total: int,
    p1_wins: int,
    p2_wins: int,
    draws: int,
    epsilon: float,
    train_enabled: bool,
    checkpoint_note: str = "",
) -> str:
    segments = [
        f"Episode {episode}/{total}",
        f"P1 Wins {ANSI.RED}{p1_wins}{ANSI.RESET}",
        f"P2 Wins {ANSI.YELLOW}{p2_wins}{ANSI.RESET}",
        f"Draws {ANSI.CYAN}{draws}{ANSI.RESET}",
        f"Epsilon {ANSI.MAGENTA}{epsilon:.4f}{ANSI.RESET}",
        f"Train {ANSI.GREEN if train_enabled else ANSI.RED}{train_enabled}{ANSI.RESET}",
    ]
    if checkpoint_note:
        segments.append(checkpoint_note)
    return " | ".join(segments)


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

        if watch_game and watch_steps > 0 and board.turn % watch_steps == 0:
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


def checkpoint_path(config: Config, episode: int) -> str:
    return os.path.join(config["save_dir"], f"{config['run_name']}_checkpoint_{episode}.pt")


def save_checkpoint(agent: Agent, config: Config, episode: int) -> str:
    os.makedirs(config["save_dir"], exist_ok=True)
    path = checkpoint_path(config, episode)
    torch.save(agent.model.state_dict(), path)
    return path


def run_training(config: Config) -> None:
    agent = Agent(
        layers=config["layers"],
        lr=config["lr"],
        epsilon=config["epsilon"],
        epsilon_decay=config["epsilon_decay"],
        epsilon_min=config["epsilon_min"],
        gamma=config["gamma"],
    )

    p1_wins = 0
    p2_wins = 0
    draws = 0

    for episode in range(1, config["num_episodes"] + 1):
        winner = play_game(
            agent=agent,
            train=config["train"],
            watch_game=config["watch_game"],
            watch_steps=config["watch_steps"],
            watch_delay=config["watch_delay"],
        )

        if winner == 1:
            p1_wins += 1
        elif winner == 2:
            p2_wins += 1
        else:
            draws += 1

        checkpoint_note = ""
        if config["save_enabled"] and episode % config["save_interval"] == 0:
            saved_path = save_checkpoint(agent, config, episode)
            checkpoint_note = f"Checkpoint {saved_path}"

        if episode % config["progress_interval"] == 0 or checkpoint_note:
            print(
                format_progress_line(
                    episode=episode,
                    total=config["num_episodes"],
                    p1_wins=p1_wins,
                    p2_wins=p2_wins,
                    draws=draws,
                    epsilon=agent.epsilon,
                    train_enabled=config["train"],
                    checkpoint_note=checkpoint_note,
                )
            )

    print(
        "\n"
        + format_progress_line(
            episode=config["num_episodes"],
            total=config["num_episodes"],
            p1_wins=p1_wins,
            p2_wins=p2_wins,
            draws=draws,
            epsilon=agent.epsilon,
            train_enabled=config["train"],
        )
    )
    print("Training complete.")

    if config["save_final"]:
        final_dir = os.path.dirname(config["final_model_path"])
        if final_dir:
            os.makedirs(final_dir, exist_ok=True)
        torch.save(agent.model.state_dict(), config["final_model_path"])
        print(f"Final model saved: {config['final_model_path']}")


if __name__ == "__main__":
    runtime_config = build_runtime_config()
    run_training(runtime_config)
