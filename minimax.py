"""
minimax.py — Minimax + alpha-beta pruning opponent for curriculum training.

Replaces mcts.py entirely. Key differences:
  - No random simulations — searches the exact game tree
  - Mutates one board in place with undo_move (zero cloning)
  - Alpha-beta pruning cuts the effective tree from 7^d to ~7^(d/2)
  - Deterministic and much stronger than MCTS at equal or lesser cost

Curriculum stages map directly to search depth:
  Stage 0 → depth 1  (sees immediate wins/blocks only)
  Stage 1 → depth 2
  Stage 2 → depth 3
  Stage 3 → depth 4  (strong tactical play)
  Stage 4 → depth 5
  Stage 5 → depth 6  (near-perfect within horizon)

Board representation: flat bytearray, same FastBoard from old mcts.py,
kept here so train.py only needs to change the import line.
"""

import random
from board import Board

ROWS = 6
COLS = 7
SIZE = ROWS * COLS

WIN_SCORE  =  1_000_000
LOSE_SCORE = -1_000_000

# ---------------------------------------------------------------------------
# Precompute win-line indices (same as before)
# ---------------------------------------------------------------------------

def _build_indices():
    lines = []
    for r in range(ROWS):
        for c in range(COLS):
            if c + 3 < COLS:
                lines.append((r*COLS+c, r*COLS+c+1, r*COLS+c+2, r*COLS+c+3))
            if r + 3 < ROWS:
                lines.append((r*COLS+c, (r+1)*COLS+c, (r+2)*COLS+c, (r+3)*COLS+c))
            if r + 3 < ROWS and c + 3 < COLS:
                lines.append((r*COLS+c, (r+1)*COLS+c+1, (r+2)*COLS+c+2, (r+3)*COLS+c+3))
            if r + 3 < ROWS and c - 3 >= 0:
                lines.append((r*COLS+c, (r+1)*COLS+c-1, (r+2)*COLS+c-2, (r+3)*COLS+c-3))

    cell_lines = [[] for _ in range(SIZE)]
    for line in lines:
        for idx in line:
            cell_lines[idx].append(line)

    return lines, [tuple(cl) for cl in cell_lines]

_WIN_LINES, _CELL_WIN_LINES = _build_indices()

# Column search order: centre-out is much better for alpha-beta pruning
# because good moves (centre control) get searched first, causing more cutoffs
_COL_ORDER = [3, 2, 4, 1, 5, 0, 6]


# ---------------------------------------------------------------------------
# FastBoard — identical interface to old mcts.py so train.py just changes
# the import. Added undo_move for minimax (no cloning needed).
# ---------------------------------------------------------------------------

class FastBoard:
    __slots__ = ("cells", "heights", "turn")

    def __init__(self):
        self.cells   = bytearray(SIZE)
        self.heights = [0] * COLS
        self.turn    = 0

    @staticmethod
    def from_board(board: Board) -> "FastBoard":
        fb = FastBoard()
        p1 = board.player1_bits.tolist()
        p2 = board.player2_bits.tolist()
        cells   = fb.cells
        heights = fb.heights
        for r in range(ROWS):
            for c in range(COLS):
                if p1[r][c]:
                    idx = r * COLS + c
                    cells[idx] = 1
                    if r + 1 > heights[c]:
                        heights[c] = r + 1
                elif p2[r][c]:
                    idx = r * COLS + c
                    cells[idx] = 2
                    if r + 1 > heights[c]:
                        heights[c] = r + 1
        fb.turn = board.turn
        return fb

    def make_move(self, col: int) -> int:
        """Returns row placed, or -1 if full."""
        r = self.heights[col]
        if r >= ROWS:
            return -1
        self.cells[r * COLS + col] = (self.turn & 1) + 1
        self.heights[col] = r + 1
        self.turn += 1
        return r

    def undo_move(self, col: int) -> None:
        """Reverse the last move in column col. No allocation."""
        self.heights[col] -= 1
        self.cells[self.heights[col] * COLS + col] = 0
        self.turn -= 1

    def check_win_at(self, cell: int, player: int) -> bool:
        c     = player
        cells = self.cells
        for a, b, d, e in _CELL_WIN_LINES[cell]:
            if cells[a] == c and cells[b] == c and cells[d] == c and cells[e] == c:
                return True
        return False

    def last_player(self) -> int:
        """The player who just moved."""
        return ((self.turn - 1) & 1) + 1

    def is_full(self) -> bool:
        return self.turn >= SIZE


# ---------------------------------------------------------------------------
# Board evaluation heuristic
# Used at leaf nodes (non-terminal positions at max depth).
# Scores each win-line by how many of our pieces are in it (with none of
# theirs), giving +weight for ours and -weight for theirs.
# ---------------------------------------------------------------------------

# Score for a line containing n of our pieces and 0 opponent pieces
_LINE_SCORE = [0, 1, 10, 50, WIN_SCORE]

def _evaluate(cells: bytearray, player: int) -> int:
    opp   = 3 - player
    score = 0
    for a, b, d, e in _WIN_LINES:
        ca, cb, cd, ce = cells[a], cells[b], cells[d], cells[e]
        # Count player and opponent pieces in this line
        p_count   = (ca == player) + (cb == player) + (cd == player) + (ce == player)
        opp_count = (ca == opp)    + (cb == opp)    + (cd == opp)    + (ce == opp)
        if opp_count == 0 and p_count > 0:
            score += _LINE_SCORE[p_count]
        elif p_count == 0 and opp_count > 0:
            score -= _LINE_SCORE[opp_count]
    # Small bonus for centre column control (col 3)
    centre_col = 3
    for r in range(ROWS):
        if cells[r * COLS + centre_col] == player:
            score += 3
        elif cells[r * COLS + centre_col] == opp:
            score -= 3
    return score


# ---------------------------------------------------------------------------
# Minimax with alpha-beta pruning
# ---------------------------------------------------------------------------

def _minimax(fb: FastBoard, depth: int, alpha: int, beta: int,
             maximising: bool, root_player: int) -> int:
    """
    Returns the heuristic score of the position from root_player's perspective.
    maximising=True  when it is root_player's turn.
    maximising=False when it is the opponent's turn.
    """
    cells   = fb.cells
    heights = fb.heights

    # Terminal / depth check
    if depth == 0 or fb.is_full():
        return _evaluate(cells, root_player)

    if maximising:
        best = LOSE_SCORE - 1
        for col in _COL_ORDER:
            if heights[col] >= ROWS:
                continue
            r   = fb.make_move(col)
            idx = r * COLS + col
            cur = (fb.turn - 1 & 1) + 1   # player who just moved
            if fb.check_win_at(idx, cur):
                fb.undo_move(col)
                # Win for root_player — scale by depth so shallower wins preferred
                return WIN_SCORE + depth
            val  = _minimax(fb, depth - 1, alpha, beta, False, root_player)
            fb.undo_move(col)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break   # beta cutoff
        return best
    else:
        best = WIN_SCORE + 1
        for col in _COL_ORDER:
            if heights[col] >= ROWS:
                continue
            r   = fb.make_move(col)
            idx = r * COLS + col
            cur = (fb.turn - 1 & 1) + 1
            if fb.check_win_at(idx, cur):
                fb.undo_move(col)
                return LOSE_SCORE - depth
            val  = _minimax(fb, depth - 1, alpha, beta, True, root_player)
            fb.undo_move(col)
            if val < best:
                best = val
            if best < beta:
                beta = best
            if alpha >= beta:
                break   # alpha cutoff
        return best


# ---------------------------------------------------------------------------
# MinimaxOpponent — drop-in replacement for MCTSOpponent
# ---------------------------------------------------------------------------

class MinimaxOpponent:
    def __init__(self, depth: int = 0, n_simulations: int = 0):
        """
        depth=0  → pure random (stage 0 baseline)
        depth=N  → minimax search to depth N (stages 1–6)
        n_simulations is ignored — kept for API compatibility.
        """
        self.depth = max(0, depth)

    def select_action(self, board: Board) -> int:
        fb = FastBoard.from_board(board)
        return self._pick(fb)

    def select_action_fast(self, fb: FastBoard) -> int:
        return self._pick(fb)

    def _pick(self, fb: FastBoard) -> int:
        heights = fb.heights
        valid   = [c for c in _COL_ORDER if heights[c] < ROWS]
        if not valid:
            return 0

        # Depth 0 = pure random, no search
        if self.depth == 0:
            return random.choice(valid)

        if len(valid) == 1:
            return valid[0]

        root_player = (fb.turn & 1) + 1
        cells       = fb.cells

        # Immediate win — no need to search
        for col in valid:
            r   = heights[col]
            idx = r * COLS + col
            cells[idx] = root_player
            win = fb.check_win_at(idx, root_player)
            cells[idx] = 0
            if win:
                return col

        # Must-block immediate opponent win
        opp = 3 - root_player
        for col in valid:
            r   = heights[col]
            idx = r * COLS + col
            cells[idx] = opp
            win = fb.check_win_at(idx, opp)
            cells[idx] = 0
            if win:
                return col

        # Full minimax search
        best_score = LOSE_SCORE - 1
        best_col   = valid[0]
        alpha      = LOSE_SCORE - 1
        beta       = WIN_SCORE  + 1

        for col in valid:
            r   = fb.make_move(col)
            idx = r * COLS + col
            # Check if this move wins immediately (minimax would catch it but
            # this short-circuits the recursion for the common case)
            if fb.check_win_at(idx, root_player):
                fb.undo_move(col)
                return col
            score = _minimax(fb, self.depth - 1, alpha, beta, False, root_player)
            fb.undo_move(col)
            if score > best_score:
                best_score = score
                best_col   = col
            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                break

        return best_col


# Alias so train.py can still write `from minimax import MCTSOpponent` if needed
MCTSOpponent = MinimaxOpponent