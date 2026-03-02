"""
board.py — Optimized 2048 engine for fast RL simulation.

Board state is a flat numpy int32 array of 16 values (row-major).
- board.move(direction)              → fast mode, returns bool
- board.move(direction, track=True)  → GUI mode, returns (changed, moves, spawn_idx)

moves: list of (r_from, c_from, r_to, c_to, is_secondary)
  is_secondary=True  → this tile gets absorbed into a merge (disappears at destination)
  is_secondary=False → this tile is the merge target (shows doubled value)

spawn_idx: flat index into self.board where new tile was placed (-1 if none)
"""

import numpy as np


class Board:
    def __init__(self):
        self.board = np.zeros(16, dtype=np.int32)
        self.score: int = 0
        self.game_over: bool = False
        self._spawn_tile()
        self._spawn_tile()

    # ------------------------------------------------------------------ spawn
    def _spawn_tile(self) -> int:
        """Place a new 2 (90%) or 4 (10%) on a random empty cell.
        Returns the flat index of the spawned tile, or -1 if board is full."""
        empty = np.flatnonzero(self.board == 0)
        if not len(empty):
            return -1
        pos = int(np.random.choice(empty))
        self.board[pos] = 2 if np.random.random() < 0.9 else 4
        return pos

    # ------------------------------------------------------------------ merge
    @staticmethod
    def _merge_line(line: np.ndarray):
        """
        Merge a 4-element int32 row leftward.
        Returns (merged_array, score_delta, moves)
        moves: [(src_col, dst_col, is_secondary), ...]
        """
        src_cols = np.flatnonzero(line)
        vals = line[src_cols]

        result = np.zeros(4, dtype=np.int32)
        moves = []
        score = 0
        dst = 0
        i = 0

        while i < len(vals):
            if i + 1 < len(vals) and vals[i] == vals[i + 1]:
                val = int(vals[i]) << 1        # x*2 via bit shift
                result[dst] = val
                score += val
                moves.append((int(src_cols[i]),     dst, False))  # stays, doubled
                moves.append((int(src_cols[i + 1]), dst, True))   # absorbed
                i += 2
            else:
                result[dst] = int(vals[i])
                moves.append((int(src_cols[i]), dst, False))
                i += 1
            dst += 1

        return result, score, moves

    # ------------------------------------------------------------------ move
    def move(self, direction: str, track: bool = False):
        """
        Execute one move.

        Fast mode  (track=False): returns bool (board changed?)
        GUI  mode  (track=True):  returns (changed, anim_moves, spawn_idx)
          anim_moves: [(r_from, c_from, r_to, c_to, is_secondary), ...]
          spawn_idx:  flat index of newly placed tile (-1 if board unchanged)
        """
        if self.game_over:
            return (False, [], -1) if track else False

        g = self.board.reshape(4, 4)

        # Normalise: all directions become "merge left on rows of `work`"
        if direction == "left":
            work = g.copy()
        elif direction == "right":
            work = g[:, ::-1].copy()
        elif direction == "up":
            work = g.T.copy()
        elif direction == "down":
            work = g.T[:, ::-1].copy()
        else:
            raise ValueError(f"Invalid direction: '{direction}'. "
                             "Use 'up', 'down', 'left', or 'right'.")

        new_work = np.empty((4, 4), dtype=np.int32)
        raw_moves = []   # (work_row, src_work_col, dst_work_col, is_secondary)
        total_score = 0

        for row in range(4):
            merged, s, row_moves = self._merge_line(work[row])
            new_work[row] = merged
            total_score += s
            if track:
                for sc, dc, is_sec in row_moves:
                    raw_moves.append((row, sc, dc, is_sec))

        # Reverse the normalisation transform
        if direction == "left":
            new_g = new_work
        elif direction == "right":
            new_g = new_work[:, ::-1]
        elif direction == "up":
            new_g = new_work.T
        elif direction == "down":
            new_g = new_work[:, ::-1].T

        new_flat = np.ascontiguousarray(new_g).ravel().astype(np.int32)
        changed = not np.array_equal(self.board, new_flat)

        spawn_idx = -1
        if changed:
            self.board = new_flat
            self.score += total_score
            spawn_idx = self._spawn_tile()

        self.game_over = not self._can_move()

        if not track:
            return changed

        # Map raw (work-space) moves back to real board coordinates
        anim_moves = []
        for wr, wsc, wdc, is_sec in raw_moves:
            if direction == "left":
                anim_moves.append((wr,       wsc,   wr,       wdc,   is_sec))
            elif direction == "right":
                anim_moves.append((wr,  3-wsc,  wr,  3-wdc,  is_sec))
            elif direction == "up":
                anim_moves.append((wsc,  wr,   wdc,   wr,    is_sec))
            elif direction == "down":
                anim_moves.append((3-wsc, wr,  3-wdc, wr,    is_sec))

        return changed, anim_moves, spawn_idx

    # ------------------------------------------------------------------ utils
    def _can_move(self) -> bool:
        if np.any(self.board == 0):
            return True
        g = self.board.reshape(4, 4)
        return (np.any(g[:, :-1] == g[:, 1:]) or # type: ignore
                np.any(g[:-1, :] == g[1:, :]))

    def reset(self):
        self.board[:] = 0
        self.score = 0
        self.game_over = False
        self._spawn_tile()
        self._spawn_tile()

    def clone(self) -> "Board":
        """Shallow-copy the board for tree search / MCTS."""
        b = Board.__new__(Board)
        b.board = self.board.copy()
        b.score = self.score
        b.game_over = self.game_over
        return b

    def get_state(self) -> np.ndarray:
        """Return a copy of the flat board for RL feature extraction."""
        return self.board.copy()

    def get_state_2d(self) -> np.ndarray:
        """Return a 4×4 view (copy) for CNN-based agents."""
        return self.board.reshape(4, 4).copy()

    def get_tensor(self) -> np.ndarray:
        """
        Return board as a (16, 4, 4) float32 numpy array ready to wrap in
        torch.from_numpy() — zero-copy when passed to the agent.

        Encoding: channel i is 1.0 wherever the tile value == 2^i.
          channel 0  → empty cells
          channel 1  → cells with value 2   (2^1)
          channel 2  → cells with value 4   (2^2)
          ...
          channel 15 → cells with value 32768 (2^15)

        One-hot over log2(value) is the standard encoding for 2048 CNNs — it
        lets the network treat each power of two as a semantically distinct
        feature rather than a raw magnitude that explodes to 2048+.
        """
        out  = np.zeros((16, 4, 4), dtype=np.float32)
        log2 = np.zeros(16, dtype=np.int64)
        mask = self.board > 0
        log2[mask] = np.log2(self.board[mask]).astype(np.int64)
        np.clip(log2, 0, 15, out=log2)
        rows = np.arange(16) // 4     # spatial row per flat index
        cols = np.arange(16) % 4      # spatial col per flat index
        out[log2, rows, cols] = 1.0
        return out

    def valid_actions(self) -> list[int]:
        """
        Return indices (0=up,1=down,2=left,3=right) of moves that change the board.
        Much cheaper than cloning: we just test each direction without spawning.
        """
        DIRS = ["up", "down", "left", "right"]
        valid = []
        g = self.board.reshape(4, 4)
        for idx, d in enumerate(DIRS):
            if d == "left":
                work = g.copy()
            elif d == "right":
                work = g[:, ::-1].copy()
            elif d == "up":
                work = g.T.copy()
            else:  # down
                work = g.T[:, ::-1].copy()
            new_work = np.empty((4, 4), dtype=np.int32)
            for row in range(4):
                new_work[row], _, _ = self._merge_line(work[row])
            # reverse transform to compare
            if d == "left":
                new_g = new_work
            elif d == "right":
                new_g = new_work[:, ::-1]
            elif d == "up":
                new_g = new_work.T
            else:
                new_g = new_work[:, ::-1].T
            if not np.array_equal(g, new_g):
                valid.append(idx)
        return valid

    @property
    def grid(self) -> np.ndarray:
        """Live 4×4 view (no copy — read-only)."""
        return self.board.reshape(4, 4)

    # ------------------------------------------------------------------ dunder
    def __str__(self) -> str:
        g = self.board.reshape(4, 4)
        col_w = max(len(str(v)) for v in self.board if v) if self.board.any() else 1
        rows = []
        for row in g:
            rows.append("  ".join(str(int(v)).rjust(col_w) if v else ".".rjust(col_w)
                                  for v in row))
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"Board(score={self.score}, game_over={self.game_over})\n{self}"
