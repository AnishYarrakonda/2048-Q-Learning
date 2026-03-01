# imports
import os
import random
import time
import statistics
import tkinter as tk
from typing import TypedDict

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
    win_reward: float
    loss_reward: float
    random_start_mode: bool
    run_name: str
    save_dir: str
    save_interval: int
    resume_training: bool
    resume_model_path: str


DEFAULT_CONFIG: Config = {
    "lr": 0.001,
    "epsilon": 1.0,
    "epsilon_decay": 0.997,
    "epsilon_min": 0.05,
    "gamma": 0.97,
    "num_episodes": 10000,
    "train": True,
    "watch_game": False,
    "watch_steps": 100,
    "watch_delay": 0.1,
    "win_reward": 0.5,
    "loss_reward": -2.0,
    "random_start_mode": False,
    "run_name": "agent",
    "save_dir": "models",
    "save_interval": 1000,
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

    def heartbeat(self) -> None:
        if not self.enabled:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
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
    config["win_reward"] = ask_float("Reward for win", config["win_reward"])
    config["loss_reward"] = ask_float("Reward for loss", config["loss_reward"])
    config["random_start_mode"] = ask_yes_no(
        "Use random non-terminal game starts (0-20 preplaced coins)?",
        config["random_start_mode"],
    )


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


def configure_saving(config: Config) -> None:
    if section_default("Saving Settings"):
        print("Using default saving settings.")
        return
    config["run_name"] = ask_text("Run/model name (used in checkpoint filenames)", config["run_name"])
    config["save_dir"] = ask_text("Directory for checkpoints and model files", config["save_dir"])
    config["save_interval"] = ask_int("Save checkpoint every N episodes", config["save_interval"], minimum=1)


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
    agent.target_model.load_state_dict(agent.model.state_dict())


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
    block_seconds: float,
    total_seconds: float,
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
        centered("Block Time", f"{block_seconds:.1f}s", 18, f"{ANSI.GREEN}{block_seconds:.1f}s{ANSI.RESET}"),
        centered("Total Time", f"{total_seconds:.1f}s", 18, f"{ANSI.GREEN}{total_seconds:.1f}s{ANSI.RESET}"),
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
    random_start_mode: bool = False,
) -> tuple[int, int]:
    board = generate_episode_start_board(random_start_mode=random_start_mode)
    done = False
    winner = 0
    last_move_by_player: dict[int, tuple[torch.Tensor, int]] = {}
    original_epsilon = agent.epsilon
    if force_zero_epsilon:
        agent.epsilon = 0.0

    try:
        if watch_game and viewer is not None:
            viewer.render(board=board, episode=episode, done=False, winner=0)
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
                    agent.push(state, action, loss_reward, next_state, done)
                    if agent._push_count % 4 == 0:
                        agent.train_step()
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
                agent.push(state, action, reward, next_state, done)
                if agent._push_count % 4 == 0:
                    agent.train_step()
                last_move_by_player[acting_player] = (state.detach().clone(), action)

                # Ensure the losing player's most recent move gets a terminal loss update.
                if done and winner in (1, 2):
                    loser = 2 if winner == 1 else 1
                    if loser in last_move_by_player:
                        loser_state, loser_action = last_move_by_player[loser]
                        agent.push(
                            loser_state,
                            loser_action,
                            loss_reward,
                            next_state,
                            True,
                        )
                        if agent._push_count % 4 == 0:
                            agent.train_step()
    finally:
        if force_zero_epsilon:
            agent.epsilon = original_epsilon

    return winner, board.turn


def clone_board_state(board: Board) -> Board:
    cloned = Board()
    cloned.player1_bits.copy_(board.player1_bits)
    cloned.player2_bits.copy_(board.player2_bits)
    cloned.turn = board.turn
    return cloned


def generate_random_non_terminal_board(max_preplaced: int = 20) -> Board:
    board = Board()
    target_moves = random.randint(0, max_preplaced)

    for _ in range(target_moves):
        valid_moves = board.valid_moves()
        if not valid_moves:
            break

        safe_moves: list[int] = []
        for col in valid_moves:
            simulated = clone_board_state(board)
            row = simulated.make_move(col)
            if row is None:
                continue
            done, _ = simulated.game_over(row=row, col=col)
            if not done:
                safe_moves.append(col)

        if not safe_moves:
            break

        chosen_col = random.choice(safe_moves)
        board.make_move(chosen_col)

    return board


def generate_episode_start_board(random_start_mode: bool) -> Board:
    if random_start_mode:
        return generate_random_non_terminal_board(max_preplaced=20)
    return Board()


def checkpoint_path(config: Config, episode: int) -> str:
    return os.path.join(config["save_dir"], f"{config['run_name']}_checkpoint_{episode}.pt")


def save_checkpoint(agent: Agent, config: Config, episode: int) -> str:
    os.makedirs(config["save_dir"], exist_ok=True)
    path = checkpoint_path(config, episode)
    torch.save(agent.model.state_dict(), path)
    return path


def run_training(config: Config) -> None:
    REPORT_INTERVAL = 50
    CHECKPOINT_INTERVAL = config["save_interval"]

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

    # Always save the original model at episode 0.
    initial_path = save_checkpoint(agent, config, 0)
    print(f"Saved checkpoint: {initial_path}")

    p1_wins = 0
    p2_wins = 0
    draws = 0
    game_lengths: list[int] = []
    viewer: TrainingTkViewer | None = None
    if config["watch_game"]:
        viewer = TrainingTkViewer(watch_delay=config["watch_delay"])
    total_start = time.perf_counter()
    block_start = total_start

    for episode in range(1, config["num_episodes"] + 1):
        if viewer is not None:
            viewer.heartbeat()
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
            random_start_mode=config["random_start_mode"],
        )

        # Visualization-only rollout (not part of training/stats): greedy policy, no updates.
        if render_this_episode and viewer is not None:
            if not viewer.enabled:
                viewer = TrainingTkViewer(watch_delay=config["watch_delay"])
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
                random_start_mode=config["random_start_mode"],
            )
        game_lengths.append(game_length)

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
        if episode % CHECKPOINT_INTERVAL == 0:
            saved_path = save_checkpoint(agent, config, episode)
            checkpoint_note = f"Checkpoint {saved_path}"
            print(f"Saved checkpoint: {saved_path}")

        avg_len = sum(game_lengths) / len(game_lengths)
        median_len = float(statistics.median(game_lengths))
        now = time.perf_counter()
        block_seconds = now - block_start
        total_seconds = now - total_start

        if episode % REPORT_INTERVAL == 0:
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
                    block_seconds=block_seconds,
                    total_seconds=total_seconds,
                    checkpoint_note=checkpoint_note,
                )
            )
            block_start = now

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
            block_seconds=time.perf_counter() - block_start,
            total_seconds=time.perf_counter() - total_start,
        )
    )
    print("Training complete.")


if __name__ == "__main__":
    runtime_config = build_runtime_config()
    run_training(runtime_config)
