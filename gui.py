"""
gui.py — Pixel-perfect 2048 with smooth animations.

Animation pipeline per move:
  1. Slide   (SLIDE_MS)  — tiles glide from old → new position (ease-out cubic)
  2. Pop     (POP_MS)    — merged tiles scale 1→1.2→1 (bounce)
  3. Spawn   (SPAWN_MS)  — new tile scales from 0→1.1→1 (ease-out)

Input feel: pressing a key while animating instantly snaps the current phase to
its end-state and starts the new move immediately — identical to the real 2048.
A small queue absorbs burst inputs so no keypress is ever lost.
"""

import tkinter as tk
import time
from collections import deque
from board import Board

# ─────────────────────────────── layout constants ─────────────────────────────
CELL    = 107          # tile side length in px
PAD     = 15           # gap between tiles / border
RADIUS  = 6            # visual corner-rounding hint (drawn as colour overlap)
W       = 4 * CELL + 5 * PAD   # canvas width == height

# ─────────────────────────────── timing ───────────────────────────────────────
SLIDE_MS = 110
POP_MS   =  90
SPAWN_MS = 160
FRAME_MS =  14    # ~70 fps

# ─────────────────────────────── colours ──────────────────────────────────────
C_WINDOW = "#faf8ef"
C_GRID   = "#bbada0"
C_EMPTY  = "#cdc1b4"

TILE_BG: dict = {
    0:    "#cdc1b4",
    2:    "#eee4da",
    4:    "#ede0c8",
    8:    "#f2b179",
    16:   "#f59563",
    32:   "#f67c5f",
    64:   "#f65e3b",
    128:  "#edcf72",
    256:  "#edcc61",
    512:  "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
}
TILE_FG: dict = {
    2: "#776e65", 4: "#776e65",
}
FG_LIGHT = "#f9f6f2"
FG_DARK  = "#776e65"

DIR_MAP = {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}


# ─────────────────────────────── helpers ──────────────────────────────────────
def _tile_bg(v: int) -> str:
    return TILE_BG.get(v, "#3c3a32")

def _tile_fg(v: int) -> str:
    return TILE_FG.get(v, FG_LIGHT)

def _font_size(v: int) -> int:
    if v <   100: return 52
    if v <  1000: return 40
    if v < 10000: return 30
    return 24

def _cell_center(r: int, c: int) -> tuple[int, int]:
    x = PAD + c * (CELL + PAD) + CELL // 2
    y = PAD + r * (CELL + PAD) + CELL // 2
    return x, y

def _ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3

def _bounce(t: float) -> float:
    """1 → 1.2 → 1 scale envelope"""
    t = max(0.0, min(1.0, t))
    return 1.0 + 0.2 * (1.0 - abs(2.0 * t - 1.0))

def _spawn_scale(t: float) -> float:
    """0 → 1.1 → 1 scale envelope"""
    t = max(0.0, min(1.0, t))
    if t < 0.75:
        return (t / 0.75) * 1.1
    return 1.1 - 0.1 * ((t - 0.75) / 0.25)


# ─────────────────────────────── GUI class ────────────────────────────────────
class GUI:
    def __init__(self):
        self.board      = Board()
        self._animating = False
        self._queue: deque[str] = deque(maxlen=8)  # pending directions
        self._snap  = False   # True → current anim phase should jump to t=1 now

        # ── window ──────────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("2048")
        self.root.configure(bg=C_WINDOW)
        self.root.resizable(False, False)

        self._build_header()

        self.canvas = tk.Canvas(
            self.root, width=W, height=W,
            bg=C_GRID, highlightthickness=0, bd=0,
        )
        self.canvas.pack(padx=10, pady=(0, 10))

        # Draw permanent empty-cell grid (never deleted)
        for r in range(4):
            for c in range(4):
                cx, cy = _cell_center(r, c)
                h = CELL // 2
                self.canvas.create_rectangle(
                    cx-h, cy-h, cx+h, cy+h,
                    fill=C_EMPTY, outline="", tags="bg",
                )

        self.root.bind("<Key>", self._on_key)
        self._redraw_tiles()
        self.root.mainloop()

    # ── header ──────────────────────────────────────────────────────────────
    def _build_header(self):
        frm = tk.Frame(self.root, bg=C_WINDOW)
        frm.pack(fill="x", padx=10, pady=(10, 5))

        tk.Label(
            frm, text="2048",
            font=("Helvetica Neue", 52, "bold"),
            fg=FG_DARK, bg=C_WINDOW,
        ).pack(side="left")

        sf = tk.Frame(frm, bg=C_GRID, padx=16, pady=5)
        sf.pack(side="right", padx=(0, 5))
        tk.Label(sf, text="SCORE", font=("Helvetica", 11, "bold"),
                 fg="#eee4da", bg=C_GRID).pack()
        self._score_var = tk.StringVar(value="0")
        tk.Label(sf, textvariable=self._score_var,
                 font=("Helvetica", 22, "bold"), fg="white", bg=C_GRID).pack()

    # ── tile drawing primitives ──────────────────────────────────────────────
    def _draw_tile(self, cx: int, cy: int, val: int,
                   scale: float = 1.0, tag: str = "tile"):
        h = max(1, int(CELL * scale / 2))
        self.canvas.create_rectangle(
            cx-h, cy-h, cx+h, cy+h,
            fill=_tile_bg(val), outline="", tags=tag,
        )
        if val and scale > 0.05:
            self.canvas.create_text(
                cx, cy,
                text=str(val),
                font=("Helvetica", max(8, int(_font_size(val) * scale)), "bold"),
                fill=_tile_fg(val),
                tags=tag,
            )

    def _redraw_tiles(self):
        """Full static redraw of current board — used between animations."""
        self.canvas.delete("tile")
        self.canvas.delete("anim")
        self.canvas.delete("pop")
        self.canvas.delete("spawn")
        g = self.board.board.reshape(4, 4)
        for r in range(4):
            for c in range(4):
                v = int(g[r, c])
                if v:
                    cx, cy = _cell_center(r, c)
                    self._draw_tile(cx, cy, v)

    # ── input handling ───────────────────────────────────────────────────────
    def _on_key(self, event):
        if event.keysym not in DIR_MAP or self.board.game_over:
            return
        d = DIR_MAP[event.keysym]
        if self._animating:
            self._snap = True        # cut current phase short immediately
            self._queue.append(d)
        else:
            self._execute_move(d)

    # ── move pipeline ────────────────────────────────────────────────────────
    def _execute_move(self, direction: str):
        old_g = self.board.board.reshape(4, 4).copy()
        result = self.board.move(direction, track=True)

        if not result[0]:              # type: ignore - board unchanged — ignore
            return

        changed, anim_moves, spawn_idx = result # type: ignore
        new_g = self.board.board.reshape(4, 4).copy()
        self._score_var.set(str(self.board.score))

        self._animating = True
        self._snap = False   # fresh flag for this animation chain
        self._phase_slide(old_g, new_g, anim_moves, spawn_idx)

    # ── phase 1 : slide ──────────────────────────────────────────────────────
    def _phase_slide(self, old_g, new_g, anim_moves, spawn_idx):
        moving_src = {(m[0], m[1]) for m in anim_moves}

        self.canvas.delete("tile")
        self.canvas.delete("anim")

        # Static tiles that don't participate in this move
        for r in range(4):
            for c in range(4):
                v = int(old_g[r, c])
                if v and (r, c) not in moving_src:
                    cx, cy = _cell_center(r, c)
                    self._draw_tile(cx, cy, v, tag="tile")

        # Create animated sprites for each moving tile
        sprites = []
        for rf, cf, rt, ct, is_sec in anim_moves:
            v = int(old_g[rf, cf])
            if not v:
                continue
            sx, sy = _cell_center(rf, cf)
            ex, ey = _cell_center(rt, ct)
            h = CELL // 2
            rect = self.canvas.create_rectangle(
                sx-h, sy-h, sx+h, sy+h,
                fill=_tile_bg(v), outline="", tags="anim",
            )
            txt = self.canvas.create_text(
                sx, sy, text=str(v),
                font=("Helvetica", _font_size(v), "bold"),
                fill=_tile_fg(v), tags="anim",
            )
            sprites.append((rect, txt, sx, sy, ex, ey))

        t0     = time.perf_counter()
        dur    = SLIDE_MS / 1000.0
        canvas = self.canvas

        def _frame():
            t  = 1.0 if self._snap else (time.perf_counter() - t0) / dur
            et = _ease_out_cubic(t)
            h  = CELL // 2
            for rect, txt, sx, sy, ex, ey in sprites:
                x = sx + (ex - sx) * et
                y = sy + (ey - sy) * et
                canvas.coords(rect, x-h, y-h, x+h, y+h)
                canvas.coords(txt,  x,   y)
            if t < 1.0:
                self.root.after(FRAME_MS, _frame)
            else:
                self._phase_pop(new_g, anim_moves, spawn_idx)

        self.root.after(0, _frame)

    # ── phase 2 : merge pop ──────────────────────────────────────────────────
    def _phase_pop(self, new_g, anim_moves, spawn_idx):
        self.canvas.delete("tile")
        self.canvas.delete("anim")

        # Which destinations had two tiles merge into them?
        dest_count: dict = {}
        for _, _, rt, ct, _ in anim_moves:
            dest_count[(rt, ct)] = dest_count.get((rt, ct), 0) + 1
        merged = {k for k, v in dest_count.items() if v > 1}

        # Determine spawn cell (exclude from early draw)
        sp_r = spawn_idx // 4 if spawn_idx >= 0 else -1
        sp_c = spawn_idx %  4 if spawn_idx >= 0 else -1

        # Draw all non-merged, non-spawn tiles as static
        for r in range(4):
            for c in range(4):
                v = int(new_g[r, c])
                if v and (r, c) not in merged and (r, c) != (sp_r, sp_c):
                    cx, cy = _cell_center(r, c)
                    self._draw_tile(cx, cy, v, tag="tile")

        if not merged:
            self._phase_spawn(new_g, spawn_idx)
            return

        t0  = time.perf_counter()
        dur = POP_MS / 1000.0

        def _frame():
            t     = 1.0 if self._snap else (time.perf_counter() - t0) / dur
            scale = _bounce(t)
            self.canvas.delete("pop")
            for r, c in merged:
                v = int(new_g[r, c])
                if v:
                    cx, cy = _cell_center(r, c)
                    self._draw_tile(cx, cy, v, scale=scale, tag="pop")
            if t < 1.0:
                self.root.after(FRAME_MS, _frame)
            else:
                self.canvas.delete("pop")
                for r, c in merged:
                    v = int(new_g[r, c])
                    if v and (r, c) != (sp_r, sp_c):
                        cx, cy = _cell_center(r, c)
                        self._draw_tile(cx, cy, v, tag="tile")
                self._phase_spawn(new_g, spawn_idx)

        self.root.after(0, _frame)

    # ── phase 3 : spawn ──────────────────────────────────────────────────────
    def _phase_spawn(self, new_g, spawn_idx: int):
        if spawn_idx < 0:
            self._anim_done()
            return

        sp_r, sp_c = spawn_idx // 4, spawn_idx % 4
        sp_v       = int(new_g[sp_r, sp_c])
        cx, cy     = _cell_center(sp_r, sp_c)
        h          = CELL // 2
        t0         = time.perf_counter()
        dur        = SPAWN_MS / 1000.0

        def _frame():
            t     = 1.0 if self._snap else (time.perf_counter() - t0) / dur
            scale = _spawn_scale(t)
            self.canvas.delete("spawn")
            # Blank the cell first so empty bg shows during early scale-in
            self.canvas.create_rectangle(
                cx-h, cy-h, cx+h, cy+h,
                fill=C_EMPTY, outline="", tags="spawn",
            )
            self._draw_tile(cx, cy, sp_v, scale=scale, tag="spawn")
            if t < 1.0:
                self.root.after(FRAME_MS, _frame)
            else:
                self.canvas.delete("spawn")
                self._draw_tile(cx, cy, sp_v, tag="tile")
                self._anim_done()

        self.root.after(0, _frame)

    # ── animation complete ───────────────────────────────────────────────────
    def _anim_done(self):
        self._animating = False
        self._snap = False
        if self.board.game_over:
            self._show_game_over()
        elif self._queue:
            self._execute_move(self._queue.popleft())

    # ── game-over overlay ────────────────────────────────────────────────────
    def _show_game_over(self):
        # Semi-transparent warm overlay
        self.canvas.create_rectangle(
            0, 0, W, W,
            fill="#f9f6f2", stipple="gray50",
            tags="gameover",
        )
        self.canvas.create_text(
            W // 2, W // 2 - 35,
            text="Game over!",
            font=("Helvetica Neue", 44, "bold"),
            fill=FG_DARK, tags="gameover",
        )
        self.canvas.create_text(
            W // 2, W // 2 + 25,
            text=f"Score: {self.board.score}",
            font=("Helvetica", 26),
            fill=FG_DARK, tags="gameover",
        )
        # New-game button
        btn = tk.Button(
            self.root, text="New Game",
            font=("Helvetica", 14, "bold"),
            fg="white", bg=C_GRID,
            activebackground="#9c8b7e",
            relief="flat", padx=20, pady=8,
            command=self._new_game,
        )
        self.canvas.create_window(W // 2, W // 2 + 85, window=btn, tags="gameover")

    def _new_game(self):
        self.canvas.delete("gameover")
        self.board.reset()
        self._score_var.set("0")
        self._redraw_tiles()


if __name__ == "__main__":
    GUI()