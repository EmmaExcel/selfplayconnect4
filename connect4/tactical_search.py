"""Bounded alpha-beta search used for the stronger web difficulty levels."""

from functools import lru_cache

from connect4.bitboard import has_won


ORDER = (3, 2, 4, 1, 5, 0, 6)
WINDOWS = []
for row in range(6):
    for col in range(7):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if 0 <= row + 3 * dr < 6 and 0 <= col + 3 * dc < 7:
                WINDOWS.append(sum(1 << ((col + i * dc) * 7 + 5 - (row + i * dr)) for i in range(4)))


def choose_move(board: list[list[int]], player: int, depth: int = 6) -> int:
    """Search from a legal position and return a legal column."""
    bits = [0, 0, 0]
    heights = [0] * 7
    for col in range(7):
        gap_seen = False
        for row in range(5, -1, -1):
            piece = board[row][col]
            if piece:
                if gap_seen:
                    raise ValueError("Board has a floating piece")
                bits[piece] |= 1 << (col * 7 + heights[col])
                heights[col] += 1
            else:
                gap_seen = True
    if not any(h < 6 for h in heights):
        raise ValueError("Board is full")

    @lru_cache(maxsize=300000)
    def evaluate(own: int, opp: int) -> int:
        score = 0
        for mask in WINDOWS:
            a = (own & mask).bit_count()
            b = (opp & mask).bit_count()
            if not b:
                score += (0, 1, 8, 75, 100000)[a]
            elif not a:
                score -= (0, 1, 9, 90, 100000)[b]
        return score

    def search(own: int, opp: int, hs: tuple[int, ...], remaining: int,
               alpha: int, beta: int) -> tuple[int, int | None]:
        legal = [c for c in ORDER if hs[c] < 6]
        if not legal:
            return 0, None
        if remaining == 0:
            return evaluate(own, opp), None

        # Wins are always preferable to a heuristic evaluation.
        for col in legal:
            if has_won(own | (1 << (col * 7 + hs[col]))):
                return 100000 + remaining, col

        # An immediate opponent win must be blocked. Multiple threats are lost.
        threats = [c for c in legal if has_won(opp | (1 << (c * 7 + hs[c])))]
        if len(threats) > 1:
            return -100000 - remaining, threats[0]
        if threats:
            legal = threats

        best_score, best_col = -10**9, legal[0]
        for col in legal:
            move = 1 << (col * 7 + hs[col])
            next_hs = list(hs)
            next_hs[col] += 1
            child_score, _ = search(opp, own | move, tuple(next_hs), remaining - 1,
                                    -beta, -alpha)
            score = -child_score
            if score > best_score:
                best_score, best_col = score, col
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_col

    return search(bits[player], bits[3 - player], tuple(heights), depth, -10**9, 10**9)[1]
