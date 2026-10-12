import numpy as np
from connect4.bitboard import pieces_to_bitboard, has_won, get_tactical_moves


class Connect4Game:
    ROWS = 6
    COLS = 7
    WIN_LENGTH = 4

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((3, self.ROWS, self.COLS), dtype=np.float32)

    def get_legal_moves(self, state: np.ndarray) -> np.ndarray:
        occupied = state[0] + state[1]
        return (occupied[0] == 0).astype(np.float32)

    def apply_move(self, state: np.ndarray, col: int) -> np.ndarray:
        if col < 0 or col >= self.COLS:
            raise ValueError(f"Column {col} is out of range [0, {self.COLS})")

        occupied = state[0] + state[1]
        if occupied[0, col] != 0:
            raise ValueError(f"Column {col} is full")

        new_state = state.copy()
        row = self._get_drop_row(occupied, col)
        new_state[0, row, col] = 1.0

        flipped = np.zeros_like(new_state)
        flipped[0] = new_state[1]
        flipped[1] = new_state[0]
        flipped[2] = 1.0 - new_state[2]

        return flipped

    def _get_drop_row(self, occupied: np.ndarray, col: int) -> int:
        for row in range(self.ROWS - 1, -1, -1):
            if occupied[row, col] == 0:
                return row
        raise ValueError(f"Column {col} is full")

    def check_terminal(self, state: np.ndarray) -> tuple:
        current_pieces = state[0]
        opponent_pieces = state[1]

        if self._has_four(opponent_pieces):
            return True, -1.0

        if self._has_four(current_pieces):
            return True, 1.0

        occupied = current_pieces + opponent_pieces
        if np.all(occupied[0] != 0):
            return True, 0.0

        return False, None

    def _has_four(self, pieces: np.ndarray) -> bool:
        return has_won(pieces_to_bitboard(pieces))

    def get_tactical_moves(self, state: np.ndarray) -> tuple:
        return get_tactical_moves(state)

    def canonical_state(self, state: np.ndarray, player: int) -> np.ndarray:
        if player == 1:
            return state.copy()
        flipped = np.zeros_like(state)
        flipped[0] = state[1]
        flipped[1] = state[0]
        flipped[2] = 1.0 - state[2]
        return flipped

    def mirror_state(self, state: np.ndarray) -> np.ndarray:
        mirrored = np.zeros_like(state)
        mirrored[0] = np.flip(state[0], axis=1)
