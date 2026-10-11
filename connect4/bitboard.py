import numpy as np

ROWS = 6
COLS = 7


def pieces_to_bitboard(pieces: np.ndarray) -> int:
    bb = 0
    for r in range(ROWS):
        for c in range(COLS):
            if pieces[r, c] == 1.0:
                bb |= (1 << (c * 7 + (5 - r)))
    return bb


def has_won(bb: int) -> bool:
    m = bb & (bb >> 7)
    if m & (m >> 14):
        return True
    m = bb & (bb >> 6)
    if m & (m >> 12):
        return True
    m = bb & (bb >> 8)
    if m & (m >> 16):
        return True
    m = bb & (bb >> 1)
    if m & (m >> 2):
        return True
    return False


def get_tactical_moves(state: np.ndarray) -> tuple:
    curr_bb = pieces_to_bitboard(state[0])
    opp_bb = pieces_to_bitboard(state[1])
