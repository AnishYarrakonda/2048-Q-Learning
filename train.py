# imports
import os
import time
import statistics
from typing import TypedDict

import matplotlib.pyplot as plt
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
    win_reward: float
    loss_reward: float
    save_enabled: bool
    save_interval: int
    run_name: str
    save_dir: str
    resume_training: bool
    resume_model_path: str


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
    "win_reward": 1.0,
    "loss_reward": -1.0,
    "save_enabled": True,
    "save_interval": 500,
    "run_name": "agent",
    "save_dir": "models",
    "resume_training": False,
    "resume_model_path": "",
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
    config["resume_training"] = ask_yes_no("Resume training from an existing model checkpoint?", config["resume_training"])
    if config["resume_training"]:
        config["resume_model_path"] = ask_text("Path to model checkpoint (.pt/.pth)", config["resume_model_path"])
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
    config["win_reward"] = ask_float("Reward for win", config["win_reward"])
    config["loss_reward"] = ask_float("Reward for loss", config["loss_reward"])


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


def build_runtime_config() -> Config:
    print("Connect Four Training Configuration")
    print("Press Enter on any prompt to accept that prompt's default.")
    print("For each section, type 'default' to skip prompts and keep the whole section default.")

    config: Config = DEFAULT_CONFIG.copy()
    configure_agent(config)
    configure_training(config)
    configure_visuals(config)
    configure_saving(config)

    if config["resume_training"] and config["resume_model_path"] and config["run_name"] == DEFAULT_CONFIG["run_name"]:
        base_name = os.path.splitext(os.path.basename(config["resume_model_path"]))[0]
        config["run_name"] = f"{base_name}_continued"

    return config


def infer_layers_from_state_dict(state_dict: dict[str, torch.Tensor]) -> list[int]:
    linear_weights: list[tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        if key.endswith(".weight") and value.ndim == 2:
            key_prefix = key.rsplit(".", 1)[0]
            if key_prefix.isdigit():
                linear_weights.append((int(key_prefix), value))

    linear_weights.sort(key=lambda item: item[0])
    if len(linear_weights) < 2:
        raise ValueError("Checkpoint does not contain enough linear layers to infer architecture.")

    hidden_layers = [int(weight.shape[0]) for _, weight in linear_weights[:-1]]
    return hidden_layers


def load_weights_for_resume(agent: Agent, model_path: str) -> None:
    loaded = torch.load(model_path, map_location=agent.device)
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        state_dict = loaded["state_dict"]
    elif isinstance(loaded, dict):
        state_dict = loaded
    else:
        raise ValueError("Unsupported checkpoint format. Expected a state_dict dictionary.")

    agent.model.load_state_dict(state_dict)


def format_progress_line(
    episode: int,
    total: int,
    p1_wins: int,
    p2_wins: int,
    draws: int,
    avg_len: float,
    median_len: float,
    epsilon: float,
    train_enabled: bool,
    checkpoint_note: str = "",
) -> str:
    def centered(label: str, value_text: str, width: int, colored_value: str) -> str:
        plain = f"{label} {value_text}"
        padded = plain.center(width)
        return padded.replace(value_text, colored_value, 1)

    segments = [
        centered("Episode", f"{episode}/{total}", 17, f"{episode}/{total}"),
        centered("P1 Wins", str(p1_wins), 14, f"{ANSI.RED}{p1_wins}{ANSI.RESET}"),
        centered("P2 Wins", str(p2_wins), 14, f"{ANSI.YELLOW}{p2_wins}{ANSI.RESET}"),
        centered("Draws", str(draws), 11, f"{ANSI.CYAN}{draws}{ANSI.RESET}"),
        centered("Avg Len", f"{avg_len:.2f}", 14, f"{ANSI.CYAN}{avg_len:.2f}{ANSI.RESET}"),
        centered("Med Len", f"{median_len:.2f}", 14, f"{ANSI.CYAN}{median_len:.2f}{ANSI.RESET}"),
        centered("Epsilon", f"{epsilon:.4f}", 16, f"{ANSI.MAGENTA}{epsilon:.4f}{ANSI.RESET}"),
        centered(
            "Train",
            str(train_enabled),
            12,
            f"{ANSI.GREEN if train_enabled else ANSI.RED}{train_enabled}{ANSI.RESET}",
        ),
    ]
    if checkpoint_note:
        segments.append(f" {checkpoint_note} ")
    return " | ".join(segments)


def play_game(
    agent: Agent,
    train: bool = True,
    watch_game: bool = False,
    watch_steps: int = 1,
    watch_delay: float = 0.2,
    win_reward: float = 1.0,
    loss_reward: float = -1.0,
) -> tuple[int, int]:
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
                agent.train_step(state, action, loss_reward, next_state, done)
            break

        done, winner = board.game_over(row, action)

        if watch_game and watch_steps > 0 and board.turn % watch_steps == 0:
            print(board)
            if watch_delay > 0:
                time.sleep(watch_delay)

        if train:
            reward = 0.0
            if done and winner == acting_player:
                reward = win_reward
            elif done and winner != 0:
                reward = loss_reward
            next_state = Board.board_to_tensor(board=board)
            agent.train_step(state, action, reward, next_state, done)

    return winner, board.turn


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    running = 0.0
    for i, val in enumerate(values):
        running += val
        if i >= window:
            running -= values[i - window]
        denom = min(i + 1, window)
        out.append(running / denom)
    return out


def build_training_plots(
    episodes: list[int],
    winners: list[int],
    game_lengths: list[int],
    epsilons: list[float],
    run_name: str,
) -> None:
    if not episodes:
        return

    window = max(10, len(episodes) // 20)

    p1_binary = [1.0 if w == 1 else 0.0 for w in winners]
    p2_binary = [1.0 if w == 2 else 0.0 for w in winners]
    draw_binary = [1.0 if w == 0 else 0.0 for w in winners]
    decisive_binary = [1.0 if w != 0 else 0.0 for w in winners]

    p1_roll = moving_average(p1_binary, window)
    p2_roll = moving_average(p2_binary, window)
    draw_roll = moving_average(draw_binary, window)
    decisive_roll = moving_average(decisive_binary, window)

    p1_edge = [p1_roll[i] - p2_roll[i] for i in range(len(p1_roll))]
    avg_len = moving_average([float(x) for x in game_lengths], window)
    med_len: list[float] = []
    for i in range(len(game_lengths)):
        start = max(0, i - window + 1)
        med_len.append(float(statistics.median(game_lengths[start : i + 1])))

    cum_p1: list[int] = []
    cum_p2: list[int] = []
    cum_draw: list[int] = []
    p1_sum = 0
    p2_sum = 0
    d_sum = 0
    for w in winners:
        if w == 1:
            p1_sum += 1
        elif w == 2:
            p2_sum += 1
        else:
            d_sum += 1
        cum_p1.append(p1_sum)
        cum_p2.append(p2_sum)
        cum_draw.append(d_sum)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training Dashboard - {run_name}", fontsize=14, fontweight="bold")

    ax1 = axes[0, 0]
    ax1.plot(episodes, cum_p1, color="red", label="Player 1 Wins")
    ax1.plot(episodes, cum_p2, color="goldenrod", label="Player 2 Wins")
    ax1.plot(episodes, cum_draw, color="steelblue", label="Draws")
    ax1.set_title("Cumulative Outcomes")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Count")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2 = axes[0, 1]
    ax2.plot(episodes, avg_len, color="purple", label=f"Avg Length ({window}-ep)")
    ax2.plot(episodes, med_len, color="teal", label=f"Median Length ({window}-ep)")
    ax2.set_title("Game Length Trends")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Moves per Game")
    ax2.grid(alpha=0.25)
    ax2.legend()

    ax3 = axes[1, 0]
    ax3.plot(episodes, p1_roll, color="red", label="P1 Win Rate")
    ax3.plot(episodes, p2_roll, color="goldenrod", label="P2 Win Rate")
    ax3.plot(episodes, draw_roll, color="steelblue", label="Draw Rate")
    ax3.plot(episodes, decisive_roll, color="green", linestyle="--", label="Decisive Rate")
    ax3.set_title(f"Rolling Outcome Rates ({window}-episode window)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Rate")
    ax3.set_ylim(-0.02, 1.02)
    ax3.grid(alpha=0.25)
    ax3.legend()

    ax4 = axes[1, 1]
    ax4.plot(episodes, epsilons, color="magenta", label="Epsilon")
    ax4.plot(episodes, p1_edge, color="black", linestyle="--", label="P1 Edge (P1 rate - P2 rate)")
    ax4.set_title("Exploration + First-Player Advantage")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Value")
    ax4.grid(alpha=0.25)
    ax4.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # type: ignore
    plt.show()


def checkpoint_path(config: Config, episode: int) -> str:
    return os.path.join(config["save_dir"], f"{config['run_name']}_checkpoint_{episode}.pt")


def save_checkpoint(agent: Agent, config: Config, episode: int) -> str:
    os.makedirs(config["save_dir"], exist_ok=True)
    path = checkpoint_path(config, episode)
    torch.save(agent.model.state_dict(), path)
    return path


def run_training(config: Config) -> None:
    layers = config["layers"]
    if config["resume_training"]:
        if not config["resume_model_path"]:
            raise ValueError("Resume training is enabled, but no resume_model_path was provided.")
        if not os.path.exists(config["resume_model_path"]):
            raise FileNotFoundError(f"Resume model not found: {config['resume_model_path']}")

        loaded = torch.load(config["resume_model_path"], map_location="cpu")
        if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            state_dict = loaded["state_dict"]
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            raise ValueError("Unsupported checkpoint format for resume.")
        layers = infer_layers_from_state_dict(state_dict)

    agent = Agent(
        layers=layers,
        lr=config["lr"],
        epsilon=config["epsilon"],
        epsilon_decay=config["epsilon_decay"],
        epsilon_min=config["epsilon_min"],
        gamma=config["gamma"],
    )

    if config["resume_training"]:
        load_weights_for_resume(agent, config["resume_model_path"])
        print(f"Resumed from checkpoint: {config['resume_model_path']}")
        print(f"Inferred layers from checkpoint: {layers}")

    p1_wins = 0
    p2_wins = 0
    draws = 0
    winners: list[int] = []
    game_lengths: list[int] = []
    episodes: list[int] = []
    epsilons: list[float] = []

    for episode in range(1, config["num_episodes"] + 1):
        winner, game_length = play_game(
            agent=agent,
            train=config["train"],
            watch_game=config["watch_game"],
            watch_steps=config["watch_steps"],
            watch_delay=config["watch_delay"],
            win_reward=config["win_reward"],
            loss_reward=config["loss_reward"],
        )
        winners.append(winner)
        game_lengths.append(game_length)
        episodes.append(episode)
        epsilons.append(agent.epsilon)

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

        avg_len = sum(game_lengths) / len(game_lengths)
        median_len = float(statistics.median(game_lengths))

        if episode % config["progress_interval"] == 0 or checkpoint_note:
            print(
                format_progress_line(
                    episode=episode,
                    total=config["num_episodes"],
                    p1_wins=p1_wins,
                    p2_wins=p2_wins,
                    draws=draws,
                    avg_len=avg_len,
                    median_len=median_len,
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
            avg_len=(sum(game_lengths) / len(game_lengths)) if game_lengths else 0.0,
            median_len=float(statistics.median(game_lengths)) if game_lengths else 0.0,
            epsilon=agent.epsilon,
            train_enabled=config["train"],
        )
    )
    print("Training complete.")
    build_training_plots(
        episodes=episodes,
        winners=winners,
        game_lengths=game_lengths,
        epsilons=epsilons,
        run_name=config["run_name"],
    )


if __name__ == "__main__":
    runtime_config = build_runtime_config()
    run_training(runtime_config)
