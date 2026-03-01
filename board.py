# import torch for tensors
import torch
from typing import Optional

# board ops stay on CPU; NN input gets moved to accelerator in board_to_tensor
_cpu = torch.device("cpu")
# MPS (Apple Silicon GPU) is SLOWER than CPU for our workload:
# batch size 64 is too small — the transfer overhead dominates compute.
# Forcing CPU gives consistent ~3s/250 games with no memory pressure spikes.
device = torch.device("cpu")

# board object
class Board:
    ROWS = 6
    COLS = 7

    # convert board state to tensor — stays on CPU (for replay buffer storage)
    @staticmethod
    def board_to_tensor_cpu(board: "Board") -> torch.Tensor:
        # Always encode from current player's perspective:
        # channel 0 = side to move, channel 1 = opponent.
        if board.turn % 2 == 0:
            current_bits = board.player1_bits
            opponent_bits = board.player2_bits
        else:
            current_bits = board.player2_bits
            opponent_bits = board.player1_bits
        return torch.stack([current_bits, opponent_bits]).flatten().unsqueeze(0)
        # NOTE: stays on CPU — ReplayBuffer.sample() moves the whole batch at once

    # convert board state to tensor on accelerator device (for inference only)
    @staticmethod
    def board_to_tensor(board: "Board") -> torch.Tensor:
        return Board.board_to_tensor_cpu(board).to(device)

    # create board
    def __init__(self: "Board") -> None:
        # stores bit boards for each player
        self.player1_bits = torch.zeros((Board.ROWS, Board.COLS), dtype=torch.float32, device=_cpu)
        self.player2_bits = torch.zeros((Board.ROWS, Board.COLS), dtype=torch.float32, device=_cpu)

        # tracks the current game turn (0 for player 1, 1 for player 2)
        self.turn = 0
        # Stores played columns in order (values 0..6), max length 42.
        self.move_history: list[int] = []


    # reset the board to the initial state
    def reset(self: "Board") -> None:
        self.player1_bits.zero_()
        self.player2_bits.zero_()
        self.turn = 0
        self.move_history.clear()


    # check horizontal 4 in a row
    def check_hor(self: "Board", x: int, y: int) -> bool:
        bits = self.player1_bits if self.turn % 2 == 1 else self.player2_bits

        start = max(y - 3, 0)
        end = min(y + 4, Board.COLS)

        row_slice = bits[x, start:end]

        for i in range(row_slice.size(0) - 3):
            if row_slice[i:i+4].all():
                return True

        return False


    # check vertical 4 in a row
    def check_vert(self: "Board", x: int, y: int) -> bool:
        bits = self.player1_bits if self.turn % 2 == 1 else self.player2_bits

        start = max(x - 3, 0)
        end = min(x + 4, Board.ROWS)

        col_slice = bits[start:end, y]

        for i in range(col_slice.size(0) - 3):
            if col_slice[i:i+4].all():
                return True

        return False


    # check main diagonal (\) 4 in a row
    def check_diag1(self: "Board", x: int, y: int) -> bool:
        bits = self.player1_bits if self.turn % 2 == 1 else self.player2_bits

        i, j = x, y
        while i > 0 and j > 0:
            i -= 1
            j -= 1

        diag_slice = []
        while i < Board.ROWS and j < Board.COLS:
            diag_slice.append(bits[i, j])
            i += 1
            j += 1

        diag_slice = torch.stack(diag_slice)

        for k in range(diag_slice.size(0) - 3):
            if diag_slice[k:k+4].all():
                return True

        return False


    # check other diagonal (/) 4 in a row
    def check_diag2(self: "Board", x: int, y: int) -> bool:
        bits = self.player1_bits if self.turn % 2 == 1 else self.player2_bits

        i, j = x, y
        while i < Board.ROWS - 1 and j > 0:
            i += 1
            j -= 1

        diag_slice = []
        while i >= 0 and j < Board.COLS:
            diag_slice.append(bits[i, j])
            i -= 1
            j += 1

        diag_slice = torch.stack(diag_slice)

        for k in range(diag_slice.size(0) - 3):
            if diag_slice[k:k+4].all():
                return True

        return False


    # full check for any winning 4 in a row
    def full_check(self: "Board", x: int, y: int) -> bool:
        return (self.check_hor(x, y) or
                self.check_vert(x, y) or
                self.check_diag1(x, y) or
                self.check_diag2(x, y))
    

    # checks if the board is full
    def is_full(self: "Board") -> bool:
        return self.turn >= Board.ROWS * Board.COLS

    # checks if a column is full
    def is_column_full(self: "Board", col: int) -> bool:
        taken = self.player1_bits[:, col] + self.player2_bits[:, col]
        return bool(taken.all().item())


    # drops a coin in the given column
    def make_move(self: "Board", col: int) -> Optional[int]:
        if col < 0 or col >= Board.COLS:
            return None

        taken = self.player1_bits[:, col] + self.player2_bits[:, col]

        if taken.all():
            return None  # Column full

        for i in range(Board.ROWS):
            if not taken[i]:
                if self.turn % 2 == 0:
                    self.player1_bits[i, col] = 1
                else:
                    self.player2_bits[i, col] = 1

                self.turn += 1
                self.move_history.append(col)
                return i  # Return the row where the coin landed

        return None
    

    # get all valid moves
    def valid_moves(self: "Board") -> list[int]:
        return [col for col in range(Board.COLS) if not self.is_column_full(col)]
    
    # returns if the game is done and the winner
    def game_over(self: "Board", row: Optional[int], col: int) -> tuple[bool, int]:
        # Defensive guard: invalid move should not crash caller.
        if row is None:
            return False, 0

        # Check if the last move resulted in a win
        if self.full_check(row, col):
            winner = 2 if self.turn % 2 == 0 else 1
            return True, winner

        if self.is_full():
            return True, 0  # Draw

        return False, 0


    # prints the board state
    def __str__(self: "Board") -> str:
        board_str = ""
        for i in range(Board.ROWS - 1, -1, -1):
            for j in range(Board.COLS):
                if self.player1_bits[i, j] == 1:
                    board_str += "X "
                elif self.player2_bits[i, j] == 1:
                    board_str += "O "
                else:
                    board_str += ". "
            board_str += "\n"
        return board_str
    

    # board representation for debugging
    def __repr__(self: "Board") -> str:
        return self.__str__()
    
# simulate a game for testing
if __name__ == "__main__":
    import time

    board = Board()
    print(board)
    moves = [1, 2, 1, 2, 1, 2, 1]
    for move in moves:
        time.sleep(0.2)
        row = board.make_move(move)
        print(board)
        if row is not None and board.full_check(row, move):
            # Determine which player just played
            winner = 2 if board.turn % 2 == 0 else 1
            print(f"Player {winner} wins!")
            break