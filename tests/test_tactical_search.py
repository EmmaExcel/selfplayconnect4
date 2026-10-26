import unittest

from connect4.tactical_search import choose_move


class TestTacticalSearch(unittest.TestCase):
    def test_takes_win_before_block(self):
        board = [[0] * 7 for _ in range(6)]
        for col in (0, 1, 2):
            board[5][col] = 1
        for col in (4, 5, 6):
            board[5][col] = 2
        self.assertEqual(choose_move(board, 1, depth=6), 3)

    def test_blocks_immediate_win(self):
        board = [[0] * 7 for _ in range(6)]
        for col in (0, 1, 2):
            board[5][col] = 2
        self.assertEqual(choose_move(board, 1, depth=6), 3)

    def test_opening_prefers_center(self):
        for depth in (6, 8, 10):
            self.assertEqual(choose_move([[0] * 7 for _ in range(6)], 1, depth=depth), 3)


if __name__ == "__main__":
    unittest.main()
