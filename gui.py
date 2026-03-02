"""
gui.py — 2048 with smooth animations, side-panel, and AI watch mode.

Layout
──────
  ┌──────────────┬─────────────────┐
  │  Side Panel  │   Board Canvas  │
  │  (controls)  │   4×4 grid      │
  └──────────────┴─────────────────┘

Side Panel
──────────
  • 2048 title + SCORE / BEST scoreboards
  • Live MOVES counter and BEST TILE this game
  • ── AI CONTROL ──
      Load Model button  (opens models/ file dialog)
      Speed slider       (0 ms → 600 ms delay between AI moves)
      ▶ Watch AI / ⏸ Pause  toggle
  • ── SESSION ──
      Games played, session best score, session best tile

Board
─────
  Larger tiles (CELL=120, PAD=16), full animation pipeline preserved.

Game Over
─────────
  Smooth fade: semi-transparent warm overlay built up over ~500 ms using
  tkinter stipple patterns (gray75 → gray50 → gray25).
  Board tiles remain partially visible through the overlay.
  Frosted panel with score, best tile, and action buttons appears on top.
"""

from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from tkinter import filedialog
import tkinter as tk

from board import Board

# ─────────────────────────────── layout ───────────────────────────────────────
CELL    = 120
PAD     = 16
W       = 4 * CELL + 5 * PAD     # board canvas side = 536
SIDE_W  = 272                     # sidebar width

# ─────────────────────────────── timing ───────────────────────────────────────
SLIDE_MS = 100
POP_MS   =  80
SPAWN_MS = 140
FRAME_MS =  14

# ─────────────────────────────── palette ──────────────────────────────────────
C_WIN    = "#faf8ef"
C_GRID   = "#bbada0"
C_EMPTY  = "#cdc1b4"
C_SIDE   = "#f0e9de"
C_PANEL  = "#e8ddd0"
C_BTN    = "#8f7a66"
C_BTN_H  = "#7a6856"
C_BTN_AI = "#f2b179"
C_BTN_AIH= "#e8a060"

TILE_BG = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8",
    8: "#f2b179", 16: "#f59563", 32: "#f67c5f",
    64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
    512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
}
TILE_FG  = {2: "#776e65", 4: "#776e65"}
FG_LIGHT = "#f9f6f2"
FG_DARK  = "#776e65"
FG_MED   = "#9c8b7e"

DIR_MAP  = {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}
DIRS     = ["up", "down", "left", "right"]

# ─────────────────────────────── helpers ──────────────────────────────────────
def _tile_bg(v):  return TILE_BG.get(v, "#3c3a32")
def _tile_fg(v):  return TILE_FG.get(v, FG_LIGHT)

def _font_size(v):
    if v <   100: return 56
    if v <  1000: return 44
    if v < 10000: return 32
    return 26

def _cell_center(r, c):
    return PAD + c*(CELL+PAD) + CELL//2,  PAD + r*(CELL+PAD) + CELL//2

def _ease_out(t):
    t = max(0., min(1., t))
    return 1. - (1.-t)**3

def _bounce(t):
    t = max(0., min(1., t))
    return 1. + 0.2*(1. - abs(2.*t - 1.))

def _spawn_s(t):
    t = max(0., min(1., t))
    return (t/0.75)*1.1 if t < 0.75 else 1.1 - 0.1*((t-0.75)/0.25)


# ─────────────────────────────── styled button helper ─────────────────────────
def _btn(parent, text, command, bg=C_BTN, fg="white", font_size=12, pady=7):
    b = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, activebackground=C_BTN_H, activeforeground="white",
        font=("Helvetica", font_size, "bold"),
        relief="flat", bd=0, cursor="hand2",
        padx=12, pady=pady,
    )
    b.bind("<Enter>", lambda e, b=b, bg=bg: b.config(bg=C_BTN_H))
    b.bind("<Leave>", lambda e, b=b, bg=bg: b.config(bg=bg))
    return b


# ─────────────────────────────── main GUI ─────────────────────────────────────
class GUI:
    def __init__(self):
        self.board       = Board()
        self._animating  = False
        self._queue: deque[str] = deque(maxlen=8)
        self._snap       = False

        # AI
        self._agent      = None
        self._model_name = ""
        self._ai_running = False
        self._ai_delay   = 200
        self._ai_job     = None

        # session stats
        self._session_games      = 0
        self._session_best_score = 0
        self._session_best_tile  = 0
        self._current_moves      = 0
        self._current_best_tile  = 0

        # root
        self.root = tk.Tk()
        self.root.title("2048")
        self.root.configure(bg=C_WIN)
        self.root.resizable(False, False)

        outer = tk.Frame(self.root, bg=C_WIN)
        outer.pack(padx=14, pady=14)

        # sidebar
        self._side = tk.Frame(outer, bg=C_SIDE, width=SIDE_W)
        self._side.pack(side="left", fill="y", padx=(0, 14))
        self._side.pack_propagate(False)

        # board canvas
        self.canvas = tk.Canvas(
            outer, width=W, height=W,
            bg=C_GRID, highlightthickness=0, bd=0,
        )
        self.canvas.pack(side="left")

        for r in range(4):
            for c in range(4):
                cx, cy = _cell_center(r, c)
                h = CELL // 2
                self.canvas.create_rectangle(
                    cx-h, cy-h, cx+h, cy+h,
                    fill=C_EMPTY, outline="", tags="bg",
                )

        self._build_sidebar()
        self.root.bind("<Key>", self._on_key)
        self._redraw_tiles()
        self.root.mainloop()

    # ─────────────────────────────── sidebar ──────────────────────────────────
    def _build_sidebar(self):
        s   = self._side
        pad = dict(padx=16)

        tk.Label(s, text="2048",
                 font=("Helvetica Neue", 58, "bold"),
                 fg=FG_DARK, bg=C_SIDE
                 ).pack(anchor="w", **pad, pady=(18, 0))

        tk.Label(s, text="Join the tiles, get to 2048!",
                 font=("Helvetica", 10), fg=FG_MED, bg=C_SIDE,
                 wraplength=SIDE_W-32
                 ).pack(anchor="w", **pad, pady=(0, 12))

        # score boxes
        row = tk.Frame(s, bg=C_SIDE)
        row.pack(fill="x", **pad)

        def _score_box(parent, label):
            f = tk.Frame(parent, bg=C_GRID, padx=14, pady=8)
            f.pack(side="left", expand=True, fill="x", padx=(0,6))
            tk.Label(f, text=label, font=("Helvetica", 10, "bold"),
                     fg="#eee4da", bg=C_GRID).pack()
            var = tk.StringVar(value="0")
            tk.Label(f, textvariable=var, font=("Helvetica", 22, "bold"),
                     fg="white", bg=C_GRID).pack()
            return var

        self._score_var = _score_box(row, "SCORE")
        self._best_var  = _score_box(row, "BEST")

        # live stats panel
        stats_f = tk.Frame(s, bg=C_PANEL, padx=14, pady=10)
        stats_f.pack(fill="x", **pad, pady=(14, 0))

        def _stat_row(parent, label):
            r2 = tk.Frame(parent, bg=C_PANEL)
            r2.pack(fill="x", pady=2)
            tk.Label(r2, text=label, font=("Helvetica", 10),
                     fg=FG_MED, bg=C_PANEL, width=12, anchor="w").pack(side="left")
            var = tk.StringVar(value="0")
            tk.Label(r2, textvariable=var, font=("Helvetica", 10, "bold"),
                     fg=FG_DARK, bg=C_PANEL).pack(side="left")
            return var

        self._moves_var    = _stat_row(stats_f, "Moves")
        self._cur_tile_var = _stat_row(stats_f, "Best tile")

        def _divider(label=""):
            f2 = tk.Frame(s, bg=C_SIDE)
            f2.pack(fill="x", **pad, pady=(16, 6))
            if label:
                tk.Label(f2, text=f"── {label} ──",
                         font=("Helvetica", 9, "bold"),
                         fg=FG_MED, bg=C_SIDE).pack(anchor="w")
            else:
                tk.Frame(f2, bg=C_PANEL, height=1).pack(fill="x")

        _divider("AI CONTROL")

        _btn(s, "⊕  Load Model", self._load_model_dialog,
             font_size=11).pack(fill="x", **pad)

        self._model_label = tk.Label(
            s, text="no model loaded",
            font=("Helvetica", 9), fg=FG_MED, bg=C_SIDE,
            wraplength=SIDE_W-32, justify="left",
        )
        self._model_label.pack(anchor="w", **pad, pady=(4, 0))

        # speed slider
        spd = tk.Frame(s, bg=C_SIDE)
        spd.pack(fill="x", **pad, pady=(12, 0))
        tk.Label(spd, text="Speed", font=("Helvetica", 10, "bold"),
                 fg=FG_DARK, bg=C_SIDE).pack(anchor="w")
        sr = tk.Frame(spd, bg=C_SIDE)
        sr.pack(fill="x")
        tk.Label(sr, text="Fast", font=("Helvetica", 8),
                 fg=FG_MED, bg=C_SIDE).pack(side="left")
        self._speed_var = tk.IntVar(value=200)
        tk.Scale(
            sr, from_=0, to=600, orient="horizontal",
            variable=self._speed_var, showvalue=False,
            bg=C_SIDE, troughcolor=C_PANEL,
            highlightthickness=0, sliderrelief="flat",
            command=lambda v: setattr(self, "_ai_delay", int(v)),
        ).pack(side="left", fill="x", expand=True)
        tk.Label(sr, text="Slow", font=("Helvetica", 8),
                 fg=FG_MED, bg=C_SIDE).pack(side="left")

        # watch / pause button
        self._watch_btn = tk.Button(
            s, text="▶  Watch AI", command=self._toggle_ai,
            bg=C_BTN_AI, fg="white",
            activebackground=C_BTN_AIH, activeforeground="white",
            font=("Helvetica", 12, "bold"),
            relief="flat", bd=0, cursor="hand2",
            padx=12, pady=9,
        )
        self._watch_btn.pack(fill="x", **pad, pady=(10, 4))
        self._watch_btn.bind("<Enter>", lambda e: self._watch_btn.config(bg=C_BTN_AIH))
        self._watch_btn.bind("<Leave>", lambda e: self._watch_btn.config(
            bg=C_BTN_AIH if self._ai_running else C_BTN_AI))

        _btn(s, "+ New Game", self._new_game_click,
             font_size=11).pack(fill="x", **pad)

        _divider("SESSION")

        sess_f = tk.Frame(s, bg=C_PANEL, padx=14, pady=10)
        sess_f.pack(fill="x", **pad, pady=(0, 4))
        self._sess_games_var = _stat_row(sess_f, "Games")
        self._sess_score_var = _stat_row(sess_f, "Best score")
        self._sess_tile_var  = _stat_row(sess_f, "Best tile")

    # ─────────────────────────────── tile drawing ─────────────────────────────
    def _draw_tile(self, cx, cy, val, scale=1.0, tag="tile"):
        h = max(1, int(CELL * scale / 2))
        self.canvas.create_rectangle(
            cx-h, cy-h, cx+h, cy+h,
            fill=_tile_bg(val), outline="", tags=tag,
        )
        if val and scale > 0.05:
            self.canvas.create_text(
                cx, cy, text=str(val),
                font=("Helvetica", max(8, int(_font_size(val)*scale)), "bold"),
                fill=_tile_fg(val), tags=tag,
            )

    def _redraw_tiles(self):
        for tag in ("tile", "anim", "pop", "spawn"):
            self.canvas.delete(tag)
        g = self.board.board.reshape(4, 4)
        for r in range(4):
            for c in range(4):
                v = int(g[r, c])
                if v:
                    self._draw_tile(*_cell_center(r, c), v)

    # ─────────────────────────────── stats ────────────────────────────────────
    def _update_stats(self):
        self._score_var.set(f"{self.board.score:,}")
        self._moves_var.set(str(self._current_moves))
        self._cur_tile_var.set(str(self._current_best_tile))

    def _update_session(self):
        self._sess_games_var.set(str(self._session_games))
        self._sess_score_var.set(f"{self._session_best_score:,}")
        self._sess_tile_var.set(str(self._session_best_tile))

    # ─────────────────────────────── input ────────────────────────────────────
    def _on_key(self, event):
        if self._ai_running:
            return
        if event.keysym not in DIR_MAP or self.board.game_over:
            return
        d = DIR_MAP[event.keysym]
        if self._animating:
            self._snap = True
            self._queue.append(d)
        else:
            self._execute_move(d)

    # ─────────────────────────────── move pipeline ────────────────────────────
    def _execute_move(self, direction: str):
        old_g  = self.board.board.reshape(4, 4).copy()
        result = self.board.move(direction, track=True)
        if not result[0]:
            if self._ai_running:
                self.root.after(0, self._ai_step)
            return

        changed, anim_moves, spawn_idx = result
        new_g = self.board.board.reshape(4, 4).copy()

        self._current_moves     += 1
        self._current_best_tile  = max(self._current_best_tile, int(self.board.board.max()))
        self._update_stats()

        self._animating = True
        self._snap      = False
        self._phase_slide(old_g, new_g, anim_moves, spawn_idx)

    # ─────────────────────── phase 1 : slide ──────────────────────────────────
    def _phase_slide(self, old_g, new_g, anim_moves, spawn_idx):
        moving_src = {(m[0], m[1]) for m in anim_moves}
        self.canvas.delete("tile")
        self.canvas.delete("anim")

        for r in range(4):
            for c in range(4):
                v = int(old_g[r, c])
                if v and (r, c) not in moving_src:
                    self._draw_tile(*_cell_center(r, c), v, tag="tile")

        sprites = []
        for rf, cf, rt, ct, _ in anim_moves:
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

        t0  = time.perf_counter()
        dur = SLIDE_MS / 1000.0

        def _frame():
            t  = 1.0 if self._snap else (time.perf_counter()-t0)/dur
            et = _ease_out(t)
            h  = CELL // 2
            for rect, txt, sx, sy, ex, ey in sprites:
                x = sx + (ex-sx)*et
                y = sy + (ey-sy)*et
                self.canvas.coords(rect, x-h, y-h, x+h, y+h)
                self.canvas.coords(txt, x, y)
            if t < 1.0:
                self.root.after(FRAME_MS, _frame)
            else:
                self._phase_pop(new_g, anim_moves, spawn_idx)

        self.root.after(0, _frame)

    # ─────────────────────── phase 2 : pop ────────────────────────────────────
    def _phase_pop(self, new_g, anim_moves, spawn_idx):
        self.canvas.delete("tile")
        self.canvas.delete("anim")

        dest_count: dict = {}
        for _, _, rt, ct, _ in anim_moves:
            dest_count[(rt, ct)] = dest_count.get((rt, ct), 0) + 1
        merged = {k for k, v in dest_count.items() if v > 1}

        sp_r = spawn_idx // 4 if spawn_idx >= 0 else -1
        sp_c = spawn_idx %  4 if spawn_idx >= 0 else -1

        for r in range(4):
            for c in range(4):
                v = int(new_g[r, c])
                if v and (r, c) not in merged and (r, c) != (sp_r, sp_c):
                    self._draw_tile(*_cell_center(r, c), v, tag="tile")

        if not merged:
            self._phase_spawn(new_g, spawn_idx)
            return

        t0  = time.perf_counter()
        dur = POP_MS / 1000.0

        def _frame():
            t     = 1.0 if self._snap else (time.perf_counter()-t0)/dur
            scale = _bounce(t)
            self.canvas.delete("pop")
            for r, c in merged:
                v = int(new_g[r, c])
                if v:
                    self._draw_tile(*_cell_center(r, c), v, scale=scale, tag="pop")
            if t < 1.0:
                self.root.after(FRAME_MS, _frame)
            else:
                self.canvas.delete("pop")
                for r, c in merged:
                    v = int(new_g[r, c])
                    if v and (r, c) != (sp_r, sp_c):
                        self._draw_tile(*_cell_center(r, c), v, tag="tile")
                self._phase_spawn(new_g, spawn_idx)

        self.root.after(0, _frame)

    # ─────────────────────── phase 3 : spawn ──────────────────────────────────
    def _phase_spawn(self, new_g, spawn_idx):
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
            t     = 1.0 if self._snap else (time.perf_counter()-t0)/dur
            scale = _spawn_s(t)
            self.canvas.delete("spawn")
            self.canvas.create_rectangle(cx-h, cy-h, cx+h, cy+h,
                                         fill=C_EMPTY, outline="", tags="spawn")
            self._draw_tile(cx, cy, sp_v, scale=scale, tag="spawn")
            if t < 1.0:
                self.root.after(FRAME_MS, _frame)
            else:
                self.canvas.delete("spawn")
                self._draw_tile(cx, cy, sp_v, tag="tile")
                self._anim_done()

        self.root.after(0, _frame)

    # ─────────────────────────────── anim done ────────────────────────────────
    def _anim_done(self):
        self._animating = False
        self._snap      = False
        if self.board.game_over:
            self._on_game_over()
        elif self._queue:
            self._execute_move(self._queue.popleft())
        elif self._ai_running:
            self._ai_job = self.root.after(self._ai_delay, self._ai_step)

    # ─────────────────────────────── game over ────────────────────────────────
    def _on_game_over(self):
        # stop AI
        self._ai_running = False
        self._watch_btn.config(text="▶  Watch AI", bg=C_BTN_AI)
        if self._ai_job:
            self.root.after_cancel(self._ai_job)
            self._ai_job = None

        # session bookkeeping
        self._session_games      += 1
        self._session_best_score  = max(self._session_best_score, self.board.score)
        self._session_best_tile   = max(self._session_best_tile, self._current_best_tile)
        cur_best = int(self._best_var.get().replace(",", ""))
        if self.board.score > cur_best:
            self._best_var.set(f"{self.board.score:,}")
        self._update_session()

        # ── animated fade ─────────────────────────────────────────────────
        # Sequence of (stipple, delay_ms) tuples building up an overlay.
        # Board tiles remain visible through the semi-transparent layer.
        FADE = [
            ("gray75", 70),
            ("gray50", 90),
            ("gray25", 110),
        ]
        FADE_COLOR = "#f9f6f2"
        current_rect = [None]

        def _fade_step(idx):
            if current_rect[0] is not None:
                self.canvas.delete(current_rect[0])
            if idx < len(FADE):
                stipple, delay = FADE[idx]
                rid = self.canvas.create_rectangle(
                    0, 0, W, W,
                    fill=FADE_COLOR, stipple=stipple,
                    outline="", tags="gameover",
                )
                current_rect[0] = rid
                self.root.after(delay, lambda: _fade_step(idx + 1))
            else:
                # Fade done — draw final panel
                _show_panel()

        def _show_panel():
            # Solid dim overlay
            self.canvas.create_rectangle(
                0, 0, W, W,
                fill=FADE_COLOR, stipple="gray25",
                outline="", tags="gameover",
            )
            # Frosted card
            px, py = W//2, W//2
            self.canvas.create_rectangle(
                px-185, py-118, px+185, py+138,
                fill="#f9f6f2", outline=C_GRID, width=2,
                tags="gameover",
            )
            self.canvas.create_text(
                px, py-72,
                text="Game Over",
                font=("Helvetica Neue", 44, "bold"),
                fill=FG_DARK, tags="gameover",
            )
            self.canvas.create_text(
                px, py-20,
                text=f"Score:  {self.board.score:,}",
                font=("Helvetica", 21),
                fill=FG_MED, tags="gameover",
            )
            self.canvas.create_text(
                px, py+18,
                text=f"Best tile:  {self._current_best_tile}",
                font=("Helvetica", 17),
                fill=FG_MED, tags="gameover",
            )

            # buttons
            has_agent = self._agent is not None
            btn_x = px - 80 if has_agent else px

            btn_new = tk.Button(
                self.root, text="New Game",
                font=("Helvetica", 13, "bold"),
                fg="white", bg=C_BTN, relief="flat",
                padx=18, pady=7, cursor="hand2",
                command=self._new_game_click,
            )
            self.canvas.create_window(btn_x, py+88, window=btn_new, tags="gameover")

            if has_agent:
                btn_watch = tk.Button(
                    self.root, text="▶ Watch AI",
                    font=("Helvetica", 13, "bold"),
                    fg="white", bg=C_BTN_AI, relief="flat",
                    padx=18, pady=7, cursor="hand2",
                    command=self._watch_again,
                )
                self.canvas.create_window(px+80, py+88, window=btn_watch, tags="gameover")

        self.root.after(60, lambda: _fade_step(0))

    def _watch_again(self):
        self.canvas.delete("gameover")
        self._new_game(start_ai=True)

    # ─────────────────────────────── new game ─────────────────────────────────
    def _new_game_click(self):
        self.canvas.delete("gameover")
        self._new_game(start_ai=False)

    def _new_game(self, start_ai=False):
        self._ai_running = False
        if self._ai_job:
            self.root.after_cancel(self._ai_job)
            self._ai_job = None
        self._queue.clear()
        self._animating = False
        self._snap      = False

        self.board.reset()
        self._current_moves     = 0
        self._current_best_tile = 0
        self._score_var.set("0")
        self._update_stats()
        self._redraw_tiles()

        if start_ai and self._agent is not None:
            self._start_ai()

    # ─────────────────────────────── AI control ───────────────────────────────
    def _load_model_dialog(self):
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        path = filedialog.askopenfilename(
            title="Load model checkpoint",
            initialdir=str(models_dir),
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_model(path)

    def _load_model(self, path: str):
        try:
            from agent import DQNAgent
            agent = DQNAgent()
            agent.load(path)
            self._agent      = agent
            self._model_name = os.path.basename(path)
            self._model_label.config(
                text=f"✓  {self._model_name}", fg="#5a7a3a",
            )
        except Exception as exc:
            self._model_label.config(text=f"✗  {exc}", fg="#c0392b")

    def _toggle_ai(self):
        if self._agent is None:
            self._model_label.config(text="← load a model first", fg="#c0392b")
            return
        if self._ai_running:
            self._stop_ai()
        else:
            self.canvas.delete("gameover")
            self._new_game(start_ai=True)

    def _start_ai(self):
        self._ai_running = True
        self._watch_btn.config(text="⏸  Pause AI", bg=C_BTN_H)
        self._ai_step()

    def _stop_ai(self):
        self._ai_running = False
        self._watch_btn.config(text="▶  Watch AI", bg=C_BTN_AI)
        if self._ai_job:
            self.root.after_cancel(self._ai_job)
            self._ai_job = None

    def _ai_step(self):
        if not self._ai_running or self.board.game_over:
            return
        if self._animating:
            self._ai_job = self.root.after(FRAME_MS, self._ai_step)
            return

        from agent import encode
        state  = encode(self.board.board)
        valid  = self.board.valid_actions()
        action = self._agent._greedy(state, valid)
        self._execute_move(DIRS[action])
        # next step scheduled in _anim_done after animation completes


if __name__ == "__main__":
    GUI()
