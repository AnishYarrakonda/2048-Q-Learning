# imports
import os
import time
import statistics
import tkinter as tk
from typing import TypedDict

import matplotlib.pyplot as plt
import torch

from agent import Agent
from board import Board


class Config(TypedDict):
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
    matplotlib_enabled: bool
    plot_update_interval: int
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
    "lr": 0.0003,
    "epsilon": 1.0,
    "epsilon_decay": 0.997,
    "epsilon_min": 0.05,
    "gamma": 0.97,
    "num_episodes": 10000,
    "train": True,
    "watch_game": False,
    "watch_steps": 100,
    "watch_delay": 0.2,
    "matplotlib_enabled": True,
    "plot_update_interval": 25,
    "progress_interval": 25,
    "win_reward": 0.5,
    "loss_reward": -1.5,
    "save_enabled": True,
    "save_interval": 2500,
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


class TrainingTkViewer:
    CELL = 72
    PAD = 10
    BG = "#0e3d63"
    BOARD = "#1769aa"
    EMPTY = "#f2f7fb"
    P1 = "#d62828"
    P2 = "#f4c430"

    def __init__(self, watch_delay: float) -> None:
        self.watch_delay = watch_delay
        self.enabled = True
        self.root = tk.Tk()
        self.root.title("Training Game Viewer")
        self.root.resizable(False, False)
        self.status_var = tk.StringVar(value="Waiting for training...")
        tk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 11, "bold")).pack(anchor="w", padx=8, pady=6)
        width = Board.COLS * self.CELL
        height = Board.ROWS * self.CELL
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg=self.BG, highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def close(self) -> None:
        if not self.enabled:
            return
        self.enabled = False
        self.root.destroy()

    def _cell_bbox(self, row: int, col: int) -> tuple[float, float, float, float]:
        visual_row = Board.ROWS - 1 - row
        x0 = col * self.CELL + self.PAD
        y0 = visual_row * self.CELL + self.PAD
        x1 = (col + 1) * self.CELL - self.PAD
        y1 = (visual_row + 1) * self.CELL - self.PAD
        return x0, y0, x1, y1

    def render(self, board: Board, episode: int, done: bool, winner: int) -> None:
        if not self.enabled:
            return
        try:
            self.canvas.delete("all")
            w = Board.COLS * self.CELL
            h = Board.ROWS * self.CELL
            self.canvas.create_rectangle(0, 0, w, h, fill=self.BOARD, outline=self.BOARD)
            for r in range(Board.ROWS):
                for c in range(Board.COLS):
                    x0, y0, x1, y1 = self._cell_bbox(r, c)
                    player = 0
                    if board.player1_bits[r, c].item() == 1:
                        player = 1
                    elif board.player2_bits[r, c].item() == 1:
                        player = 2
                    color = self.EMPTY if player == 0 else (self.P1 if player == 1 else self.P2)
                    self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline="#1e1e1e", width=2)

            if done:
                if winner == 0:
                    status = f"Episode {episode}: Draw"
                else:
                    status = f"Episode {episode}: Player {winner} won"
            else:
                next_player = 1 if board.turn % 2 == 0 else 2
                status = f"Episode {episode}: Turn {board.turn} | Next Player {next_player}"
            self.status_var.set(status)
            self.root.update_idletasks()
            self.root.update()
            if self.watch_delay > 0:
                time.sleep(self.watch_delay)
        except tk.TclError:
            self.enabled = False


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
    config["watch_game"] = ask_yes_no("Enable Tkinter game viewer during training?", config["watch_game"])
    if config["watch_game"]:
        config["watch_steps"] = ask_int("Show a full game in Tkinter every N episodes", config["watch_steps"], minimum=1)
        config["watch_delay"] = ask_float(
            "Delay between Tkinter move updates (seconds)", config["watch_delay"], minimum=0.0
        )
    config["matplotlib_enabled"] = ask_yes_no(
        "Enable live Matplotlib dashboard during training?",
        config["matplotlib_enabled"],
    )
    if config["matplotlib_enabled"]:
        config["plot_update_interval"] = ask_int(
            "Update Matplotlib every N episodes",
            config["plot_update_interval"],
            minimum=1,
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
    viewer: TrainingTkViewer | None = None,
    episode: int = 0,
    force_zero_epsilon: bool = False,
) -> tuple[int, int]:
    board = Board()
    done = False
    winner = 0
    last_move_by_player: dict[int, tuple[torch.Tensor, int]] = {}
    original_epsilon = agent.epsilon
    if force_zero_epsilon:
        agent.epsilon = 0.0

    try:
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

            if watch_game and viewer is not None:
                viewer.render(board=board, episode=episode, done=done, winner=winner)

            if train:
                reward = 0.0
                if done and winner == acting_player:
                    reward = win_reward
                elif done and winner != 0:
                    reward = loss_reward
                next_state = Board.board_to_tensor(board=board)
                agent.train_step(state, action, reward, next_state, done)
                last_move_by_player[acting_player] = (state.detach().clone(), action)

                # Ensure the losing player's most recent move gets a terminal loss update.
                if done and winner in (1, 2):
                    loser = 2 if winner == 1 else 1
                    if loser in last_move_by_player:
                        loser_state, loser_action = last_move_by_player[loser]
                        agent.train_step(
                            loser_state,
                            loser_action,
                            loss_reward,
                            next_state,
                            True,
                        )
    finally:
        if force_zero_epsilon:
            agent.epsilon = original_epsilon

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


def update_live_training_plots(
    fig: plt.Figure, # type: ignore
    axes: list[list[plt.Axes]], # type: ignore
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
    p1_sum = p2_sum = d_sum = 0
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

    for row in axes:
        for ax in row:
            ax.clear()

    axes[0][0].plot(episodes, cum_p1, color="red", label="Player 1 Wins")
    axes[0][0].plot(episodes, cum_p2, color="goldenrod", label="Player 2 Wins")
    axes[0][0].plot(episodes, cum_draw, color="steelblue", label="Draws")
    axes[0][0].set_title("Cumulative Outcomes")
    axes[0][0].set_xlabel("Episode")
    axes[0][0].set_ylabel("Count")
    axes[0][0].grid(alpha=0.25)
    axes[0][0].legend()

    axes[0][1].plot(episodes, avg_len, color="purple", label=f"Avg Length ({window}-ep)")
    axes[0][1].plot(episodes, med_len, color="teal", label=f"Median Length ({window}-ep)")
    axes[0][1].set_title("Game Length Trends")
    axes[0][1].set_xlabel("Episode")
    axes[0][1].set_ylabel("Moves per Game")
    axes[0][1].grid(alpha=0.25)
    axes[0][1].legend()

    axes[1][0].plot(episodes, p1_roll, color="red", label="P1 Win Rate")
    axes[1][0].plot(episodes, p2_roll, color="goldenrod", label="P2 Win Rate")
    axes[1][0].plot(episodes, draw_roll, color="steelblue", label="Draw Rate")
    axes[1][0].plot(episodes, decisive_roll, color="green", linestyle="--", label="Decisive Rate")
    axes[1][0].set_title(f"Rolling Outcome Rates ({window}-episode window)")
    axes[1][0].set_xlabel("Episode")
    axes[1][0].set_ylabel("Rate")
    axes[1][0].set_ylim(-0.02, 1.02)
    axes[1][0].grid(alpha=0.25)
    axes[1][0].legend()

    axes[1][1].plot(episodes, epsilons, color="magenta", label="Epsilon")
    axes[1][1].plot(episodes, p1_edge, color="black", linestyle="--", label="P1 Edge (P1-P2 rate)")
    axes[1][1].set_title("Exploration + First-Player Advantage")
    axes[1][1].set_xlabel("Episode")
    axes[1][1].set_ylabel("Value")
    axes[1][1].grid(alpha=0.25)
    axes[1][1].legend()

    fig.suptitle(f"Training Dashboard - {run_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # type: ignore
    fig.canvas.draw_idle()
    plt.pause(0.001)


def checkpoint_path(config: Config, episode: int) -> str:
    return os.path.join(config["save_dir"], f"{config['run_name']}_checkpoint_{episode}.pt")


def save_checkpoint(agent: Agent, config: Config, episode: int) -> str:
    os.makedirs(config["save_dir"], exist_ok=True)
    path = checkpoint_path(config, episode)
    torch.save(agent.model.state_dict(), path)
    return path


def run_training(config: Config) -> None:
    if config["resume_training"]:
        if not config["resume_model_path"]:
            raise ValueError("Resume training is enabled, but no resume_model_path was provided.")
        if not os.path.exists(config["resume_model_path"]):
            raise FileNotFoundError(f"Resume model not found: {config['resume_model_path']}")

    agent = Agent(
        lr=config["lr"],
        epsilon=config["epsilon"],
        epsilon_decay=config["epsilon_decay"],
        epsilon_min=config["epsilon_min"],
        gamma=config["gamma"],
    )

    if config["resume_training"]:
        load_weights_for_resume(agent, config["resume_model_path"])
        print(f"Resumed from checkpoint: {config['resume_model_path']}")

    p1_wins = 0
    p2_wins = 0
    draws = 0
    winners: list[int] = []
    game_lengths: list[int] = []
    episodes: list[int] = []
    epsilons: list[float] = []
    viewer: TrainingTkViewer | None = None
    if config["watch_game"]:
        viewer = TrainingTkViewer(watch_delay=config["watch_delay"])

    live_fig: plt.Figure | None = None # type: ignore
    live_axes: list[list[plt.Axes]] | None = None # type: ignore
    if config["matplotlib_enabled"]:
        plt.ion()
        live_fig, live_axes_np = plt.subplots(2, 2, figsize=(15, 10))
        live_axes = live_axes_np.tolist()  # type: ignore[assignment]
        plt.show(block=False)

    for episode in range(1, config["num_episodes"] + 1):
        render_this_episode = bool(config["watch_game"] and config["watch_steps"] > 0 and episode % config["watch_steps"] == 0)
        winner, game_length = play_game(
            agent=agent,
            train=config["train"],
            watch_game=False,
            watch_steps=config["watch_steps"],
            watch_delay=config["watch_delay"],
            win_reward=config["win_reward"],
            loss_reward=config["loss_reward"],
            viewer=None,
            episode=episode,
            force_zero_epsilon=False,
        )

        # Visualization-only rollout (not part of training/stats): greedy policy, no updates.
        if render_this_episode and viewer is not None:
            play_game(
                agent=agent,
                train=False,
                watch_game=True,
                watch_steps=config["watch_steps"],
                watch_delay=config["watch_delay"],
                win_reward=config["win_reward"],
                loss_reward=config["loss_reward"],
                viewer=viewer,
                episode=episode,
                force_zero_epsilon=True,
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

        # Decay epsilon once every 10 episodes (not every move).
        if config["train"] and episode % 10 == 0 and agent.epsilon > agent.epsilon_min:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

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
        if (
            config["matplotlib_enabled"]
            and live_fig is not None
            and live_axes is not None
            and episode % config["plot_update_interval"] == 0
        ):
            update_live_training_plots(
                fig=live_fig,
                axes=live_axes,
                episodes=episodes,
                winners=winners,
                game_lengths=game_lengths,
                epsilons=epsilons,
                run_name=config["run_name"],
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
    if config["matplotlib_enabled"] and live_fig is not None and live_axes is not None:
        update_live_training_plots(
            fig=live_fig,
            axes=live_axes,
            episodes=episodes,
            winners=winners,
            game_lengths=game_lengths,
            epsilons=epsilons,
            run_name=config["run_name"],
        )
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    runtime_config = build_runtime_config()
    run_training(runtime_config)
