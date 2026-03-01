import random
import re
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import torch
import torch.nn as nn


class Board:
    """Tracks Connect Four state and win/draw logic."""

    ROWS = 6
    COLS = 7

    def __init__(self) -> None:
        self.grid: list[list[int]] = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.turn = 0

    def reset(self) -> None:
        self.grid = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.turn = 0

    def valid_moves(self) -> list[int]:
        return [c for c in range(self.COLS) if self.grid[self.ROWS - 1][c] == 0]

    def is_full(self) -> bool:
        return all(self.grid[self.ROWS - 1][c] != 0 for c in range(self.COLS))

    def drop_token(self, col: int, player: int) -> Optional[int]:
        if col < 0 or col >= self.COLS:
            return None
        for row in range(self.ROWS):
            if self.grid[row][col] == 0:
                self.grid[row][col] = player
                self.turn += 1
                return row
        return None

    def _line_cells(self, row: int, col: int, dr: int, dc: int, player: int) -> list[tuple[int, int]]:
        cells = [(row, col)]

        r, c = row + dr, col + dc
        while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.grid[r][c] == player:
            cells.append((r, c))
            r += dr
            c += dc

        r, c = row - dr, col - dc
        while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.grid[r][c] == player:
            cells.append((r, c))
            r -= dr
            c -= dc

        return cells

    def check_winner(self, row: int, col: int, player: int) -> tuple[bool, list[tuple[int, int]]]:
        # Check all four directions passing through the last token.
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            line = self._line_cells(row, col, dr, dc, player)
            if len(line) >= 4:
                # Keep only four contiguous cells centered around the last move if possible.
                line_sorted = sorted(line, key=lambda rc: (rc[0], rc[1]))
                if (row, col) in line_sorted:
                    idx = line_sorted.index((row, col))
                    start = max(0, min(idx - 3, len(line_sorted) - 4))
                    return True, line_sorted[start : start + 4]
                return True, line_sorted[:4]
        return False, []

    def game_state_after(self, row: int, col: int, player: int) -> tuple[bool, int, list[tuple[int, int]]]:
        won, cells = self.check_winner(row, col, player)
        if won:
            return True, player, cells
        if self.is_full():
            return True, 0, []
        return False, 0, []

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        state = torch.zeros((1, 3, self.ROWS, self.COLS), dtype=torch.float32, device=device)
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.grid[r][c] == 1:
                    state[0, 0, r, c] = 1.0
                elif self.grid[r][c] == 2:
                    state[0, 1, r, c] = 1.0
        # 1.0 when it's player 1 to move next, 0.0 when it's player 2.
        turn_value = 1.0 if self.turn % 2 == 0 else 0.0
        state[0, 2, :, :] = turn_value
        return state


class Agent:
    """Simple AI interface around a PyTorch model."""

    def __init__(self) -> None:
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.model: Optional[nn.Module] = None
        self.model_path: Optional[str] = None

    @staticmethod
    def _natural_key(text: str) -> list[object]:
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]

    def _build_model_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> nn.Sequential:
        weight_items: list[tuple[str, torch.Tensor]] = []
        for key, value in state_dict.items():
            if key.endswith("weight") and value.ndim == 2:
                weight_items.append((key, value))

        if not weight_items:
            raise ValueError("No linear layer weights found in state dict.")

        weight_items.sort(key=lambda kv: self._natural_key(kv[0]))

        layers: list[nn.Module] = [nn.Flatten()]
        for idx, (_, weight) in enumerate(weight_items):
            out_features, in_features = weight.shape
            layers.append(nn.Linear(in_features, out_features, device=self.device))
            if idx < len(weight_items) - 1:
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers).to(self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def load_model(self, path: str) -> tuple[bool, str]:
        try:
            loaded = torch.load(path, map_location=self.device)
            if isinstance(loaded, nn.Module):
                model = loaded.to(self.device)
                model.eval()
            elif isinstance(loaded, dict):
                if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                    state_dict = loaded["state_dict"]
                else:
                    state_dict = loaded
                model = self._build_model_from_state_dict(state_dict)
            else:
                raise ValueError("Unsupported model file format.")

            self.model = model
            self.model_path = path
            return True, f"Loaded model: {path}"
        except Exception as exc:  # pragma: no cover - GUI path
            self.model = None
            self.model_path = None
            return False, f"Failed to load model: {exc}"

    def select_move(self, board: Board, valid_moves: list[int]) -> int:
        if not valid_moves:
            return 0

        if self.model is None:
            return random.choice(valid_moves)

        with torch.no_grad():
            q_values = self.model(board.to_tensor(self.device)).flatten()

        masked = q_values.clone()
        for col in range(Board.COLS):
            if col not in valid_moves:
                masked[col] = -float("inf")

        return int(torch.argmax(masked).item())


class ConnectFourGUI:
    CELL = 92
    PADDING = 12
    BOARD_COLOR = "#1769aa"
    EMPTY_COLOR = "#f2f7fb"
    P1_COLOR = "#d62828"
    P2_COLOR = "#f4c430"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Connect Four")
        self.root.resizable(False, False)

        self.board = Board()
        self.agent = Agent()

        self.mode_var = tk.StringVar(value="Human vs AI")
        self.human_side_var = tk.StringVar(value="First (Player 1)")
        self.turn_var = tk.StringVar(value="Turn: Player 1 (Red)")
        self.result_var = tk.StringVar(value="")
        self.match_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value="Model: Not loaded (AI will pick random valid moves)")

        self.current_player = 1
        self.game_over = False
        self.game_started = False
        self.animating = False
        self.ai_job: Optional[str] = None
        self.active_animation_job: Optional[str] = None
        self.active_falling_token: Optional[int] = None
        self.animation_generation = 0

        self.token_ids: list[list[Optional[int]]] = [
            [None for _ in range(Board.COLS)] for _ in range(Board.ROWS)
        ]

        self._build_controls()
        self._build_board_canvas()
        self.draw_static_board()
        self.update_match_label()
        self.schedule_ai_if_needed()

    def _build_controls(self) -> None:
        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=8)

        tk.Label(controls, text="Mode:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        mode_menu = tk.OptionMenu(
            controls,
            self.mode_var,
            "Human vs AI",
            "Human vs Human",
            command=lambda _value: self.on_mode_change(),
        )
        mode_menu.grid(row=0, column=1, sticky="w")

        tk.Label(controls, text="Human Side:").grid(row=0, column=2, sticky="w", padx=(10, 4))
        side_menu = tk.OptionMenu(
            controls,
            self.human_side_var,
            "First (Player 1)",
            "Second (Player 2)",
            command=lambda _value: self.on_human_side_change(),
        )
        side_menu.grid(row=0, column=3, sticky="w")

        tk.Button(controls, text="Load Model (.pt)", command=self.load_model_file).grid(
            row=0, column=4, padx=8
        )
        tk.Button(controls, text="Start Game", command=self.start_game).grid(row=0, column=5, padx=4)
        tk.Button(controls, text="Reset Game", command=self.reset_game).grid(row=0, column=6, padx=4)
        tk.Button(controls, text="Quit", command=self.root.destroy).grid(row=0, column=7, padx=4)

        tk.Label(controls, text="AI Speed (ms):").grid(row=0, column=8, padx=(12, 4))
        self.ai_speed = tk.Scale(controls, from_=30, to=900, orient="horizontal", length=160)
        self.ai_speed.set(220)
        self.ai_speed.grid(row=0, column=9, sticky="w")

        tk.Label(self.root, textvariable=self.match_var, font=("Helvetica", 11)).pack(anchor="w", padx=10)
        tk.Label(self.root, textvariable=self.turn_var, font=("Helvetica", 12, "bold")).pack(anchor="w", padx=10)
        tk.Label(self.root, textvariable=self.result_var, font=("Helvetica", 11)).pack(anchor="w", padx=10, pady=(2, 2))
        tk.Label(self.root, textvariable=self.model_var, font=("Helvetica", 10), fg="#333").pack(anchor="w", padx=10)

    def _build_board_canvas(self) -> None:
        width = Board.COLS * self.CELL
        height = Board.ROWS * self.CELL
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#0e3d63", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def cell_bbox(self, row: int, col: int) -> tuple[float, float, float, float]:
        # Board row 0 is bottom. Canvas row 0 is top.
        visual_row = Board.ROWS - 1 - row
        x0 = col * self.CELL + self.PADDING
        y0 = visual_row * self.CELL + self.PADDING
        x1 = (col + 1) * self.CELL - self.PADDING
        y1 = (visual_row + 1) * self.CELL - self.PADDING
        return x0, y0, x1, y1

    def draw_static_board(self) -> None:
        self.canvas.delete("all")
        w = Board.COLS * self.CELL
        h = Board.ROWS * self.CELL
        self.canvas.create_rectangle(0, 0, w, h, fill=self.BOARD_COLOR, outline=self.BOARD_COLOR)

        # Draw empty circles (board holes).
        for r in range(Board.ROWS):
            for c in range(Board.COLS):
                x0, y0, x1, y1 = self.cell_bbox(r, c)
                self.canvas.create_oval(x0, y0, x1, y1, fill=self.EMPTY_COLOR, outline="#d9e9f7", width=2)

        # Re-draw current tokens after board repaint.
        for r in range(Board.ROWS):
            for c in range(Board.COLS):
                player = self.board.grid[r][c]
                if player != 0:
                    self.token_ids[r][c] = self.draw_token(r, c, player)

    def draw_token(self, row: int, col: int, player: int) -> int:
        color = self.P1_COLOR if player == 1 else self.P2_COLOR
        x0, y0, x1, y1 = self.cell_bbox(row, col)
        return self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline="#222", width=2)

    def on_mode_change(self) -> None:
        self.update_match_label()
        self.reset_game()

    def on_human_side_change(self) -> None:
        self.update_match_label()
        if self.mode_var.get() == "Human vs AI":
            self.reset_game()

    def reset_game(self) -> None:
        if self.ai_job is not None:
            self.root.after_cancel(self.ai_job)
            self.ai_job = None
        self.cancel_active_animation()

        self.board.reset()
        self.current_player = 1
        self.game_over = False
        self.game_started = False
        self.animating = False
        self.result_var.set("")
        self.turn_var.set("Press Start Game to begin")
        self.token_ids = [[None for _ in range(Board.COLS)] for _ in range(Board.ROWS)]
        self.draw_static_board()
        self.update_match_label()
        # Reset automatically makes the board ready for a new game.
        self.start_game()

    def start_game(self) -> None:
        if self.ai_job is not None:
            self.root.after_cancel(self.ai_job)
            self.ai_job = None
        self.cancel_active_animation()
        self.game_over = False
        self.game_started = True
        self.result_var.set("")
        if self.mode_var.get() == "Human vs AI" and self.human_side_var.get().startswith("Second"):
            self.current_player = 1
        else:
            self.current_player = 1
        color_name = "Red" if self.current_player == 1 else "Yellow"
        self.turn_var.set(f"Turn: Player {self.current_player} ({color_name})")
        self.schedule_ai_if_needed()

    def cancel_active_animation(self) -> None:
        self.animation_generation += 1
        if self.active_animation_job is not None:
            self.root.after_cancel(self.active_animation_job)
            self.active_animation_job = None
        if self.active_falling_token is not None:
            self.canvas.delete(self.active_falling_token)
            self.active_falling_token = None
        self.animating = False

    def model_display_name(self) -> str:
        if self.agent.model_path:
            return os.path.basename(self.agent.model_path)
        return "random-policy"

    def update_match_label(self) -> None:
        mode = self.mode_var.get()
        model_name = self.model_display_name()
        if mode == "Human vs Human":
            text = "Player 1: Human | Player 2: Human"
        elif self.human_side_var.get().startswith("Second"):
            text = f"Player 1: CPU ({model_name}) | Player 2: Human"
        else:
            text = f"Player 1: Human | Player 2: CPU ({model_name})"
        self.match_var.set(text)

    def load_model_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Load PyTorch Model",
            filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*.*")],
        )
        if not path:
            return

        success, msg = self.agent.load_model(path)
        self.model_var.set(
            f"Model: {path}" if success else "Model: Not loaded (AI will pick random valid moves)"
        )
        self.update_match_label()
        if success:
            messagebox.showinfo("Model", msg)
        else:
            messagebox.showerror("Model Load Error", msg)

    def on_canvas_click(self, event: tk.Event) -> None:
        if self.game_over or self.animating or not self.game_started:
            return

        if not self.is_human_turn():
            return

        col = max(0, min(Board.COLS - 1, event.x // self.CELL))
        self.try_move(col)

    def is_human_turn(self) -> bool:
        mode = self.mode_var.get()
        if mode == "Human vs Human":
            return True
        if mode == "Human vs AI":
            if self.human_side_var.get().startswith("Second"):
                return self.current_player == 2
            return self.current_player == 1
        return False

    def is_ai_turn(self) -> bool:
        mode = self.mode_var.get()
        if mode == "Human vs AI":
            if self.human_side_var.get().startswith("Second"):
                return self.current_player == 1
            return self.current_player == 2
        return False

    def schedule_ai_if_needed(self) -> None:
        if self.game_over or self.animating:
            return
        if not self.is_ai_turn():
            return

        delay = int(self.ai_speed.get())
        self.ai_job = self.root.after(delay, self.run_ai_turn)

    def run_ai_turn(self) -> None:
        self.ai_job = None
        if self.game_over or self.animating or not self.game_started or not self.is_ai_turn():
            return

        valid = self.board.valid_moves()
        if not valid:
            return

        col = self.agent.select_move(self.board, valid)
        self.try_move(col)

    def try_move(self, col: int) -> None:
        if self.animating or self.game_over or not self.game_started:
            return

        row = self.board.drop_token(col, self.current_player)
        if row is None:
            # Column full: ignore click/move as requested.
            self.result_var.set(f"Column {col + 1} is full.")
            return

        self.animating = True
        self.animate_drop(col, row, self.current_player, lambda: self.finalize_move(row, col))

    def animate_drop(self, col: int, row: int, player: int, on_done) -> None:
        """Smooth falling token animation with a small bounce on landing."""
        generation = self.animation_generation
        color = self.P1_COLOR if player == 1 else self.P2_COLOR
        x0, y0, x1, y1 = self.cell_bbox(row, col)
        diameter = x1 - x0
        cx = (x0 + x1) / 2
        radius = diameter / 2

        start_y = -radius - 10
        target_y = (y0 + y1) / 2

        token_id = self.canvas.create_oval(
            cx - radius,
            start_y - radius,
            cx + radius,
            start_y + radius,
            fill=color,
            outline="#222",
            width=2,
        )
        self.active_falling_token = token_id

        velocity = 0.0
        gravity = 2.2
        damping = 0.35
        bounce_count = 0
        max_bounces = 1

        def step(current_y: float, v: float, bounces: int) -> None:
            if generation != self.animation_generation:
                self.canvas.delete(token_id)
                return

            new_v = v + gravity
            new_y = current_y + new_v

            if new_y >= target_y:
                new_y = target_y
                if bounces < max_bounces:
                    new_v = -max(2.0, new_v * damping)
                    bounces += 1
                else:
                    self.canvas.delete(token_id)
                    self.active_falling_token = None
                    self.active_animation_job = None
                    self.token_ids[row][col] = self.draw_token(row, col, player)
                    on_done()
                    return

            self.canvas.coords(
                token_id,
                cx - radius,
                new_y - radius,
                cx + radius,
                new_y + radius,
            )
            self.active_animation_job = self.root.after(16, lambda: step(new_y, new_v, bounces))

        step(start_y, velocity, bounce_count)

    def finalize_move(self, row: int, col: int) -> None:
        player = self.current_player
        self.animating = False

        done, winner, win_cells = self.board.game_state_after(row, col, player)
        if done:
            self.game_over = True
            if winner == 1:
                self.turn_var.set("Turn: Game Over")
                self.result_var.set("Player 1 wins!")
                self.highlight_winning_cells(win_cells)
            elif winner == 2:
                self.turn_var.set("Turn: Game Over")
                self.result_var.set("Player 2 wins!")
                self.highlight_winning_cells(win_cells)
            else:
                self.turn_var.set("Turn: Game Over")
                self.result_var.set("Draw!")
            return

        self.current_player = 2 if self.current_player == 1 else 1
        color_name = "Red" if self.current_player == 1 else "Yellow"
        self.turn_var.set(f"Turn: Player {self.current_player} ({color_name})")
        self.result_var.set("")
        self.schedule_ai_if_needed()

    def highlight_winning_cells(self, cells: list[tuple[int, int]]) -> None:
        if not cells:
            return

        # Flash winning four tokens a few times.
        flash_total = 6

        def flash(step_idx: int) -> None:
            on = step_idx % 2 == 0
            for r, c in cells:
                token_id = self.token_ids[r][c]
                if token_id is not None:
                    self.canvas.itemconfig(
                        token_id,
                        outline="#ffffff" if on else "#111111",
                        width=5 if on else 2,
                    )
            if step_idx < flash_total:
                self.root.after(150, lambda: flash(step_idx + 1))

        flash(0)

def main() -> None:
    root = tk.Tk()
    app = ConnectFourGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
