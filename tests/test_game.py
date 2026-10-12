import unittest
import numpy as np
from connect4.game import Connect4Game


class TestConnect4Game(unittest.TestCase):
    def setUp(self):
        self.game = Connect4Game()

    def test_initial_state(self):
        state = self.game.get_initial_state()
        self.assertEqual(state.shape, (3, 6, 7))
        self.assertEqual(np.sum(state), 0.0)
        legal = self.game.get_legal_moves(state)
        self.assertEqual(len(legal), 7)
        self.assertTrue(np.all(legal == 1.0))

    def test_apply_move_and_drop_order(self):
        state = self.game.get_initial_state()
        next_state = self.game.apply_move(state, 3)
        self.assertEqual(next_state[1, 5, 3], 1.0)
        self.assertEqual(next_state[0, 5, 3], 0.0)
        self.assertEqual(next_state[2, 0, 0], 1.0)

        next_state_2 = self.game.apply_move(next_state, 3)
        self.assertEqual(next_state_2[0, 5, 3], 1.0)
        self.assertEqual(next_state_2[1, 4, 3], 1.0)
        self.assertEqual(next_state_2[2, 0, 0], 0.0)

    def test_column_full_legal_moves(self):
        state = self.game.get_initial_state()
        for _ in range(6):
            state = self.game.apply_move(state, 0)
        legal = self.game.get_legal_moves(state)
        self.assertEqual(legal[0], 0.0)
        self.assertTrue(np.all(legal[1:] == 1.0))
        with self.assertRaises(ValueError):
            self.game.apply_move(state, 0)

    def test_horizontal_win(self):
        state = self.game.get_initial_state()
        moves = [0, 0, 1, 1, 2, 2, 3]
        for m in moves:
            state = self.game.apply_move(state, m)
        is_term, val = self.game.check_terminal(state)
        self.assertTrue(is_term)
        self.assertEqual(val, -1.0)

    def test_vertical_win(self):
        state = self.game.get_initial_state()
        moves = [0, 1, 0, 1, 0, 1, 0]
        for m in moves:
            state = self.game.apply_move(state, m)
        is_term, val = self.game.check_terminal(state)
        self.assertTrue(is_term)
        self.assertEqual(val, -1.0)

    def test_diagonal_up_win(self):
        state = self.game.get_initial_state()
        moves = [0, 1, 1, 2, 2, 3, 2, 3, 3, 0, 3]
        for m in moves:
            state = self.game.apply_move(state, m)
        is_term, val = self.game.check_terminal(state)
        self.assertTrue(is_term)
        self.assertEqual(val, -1.0)

    def test_diagonal_down_win(self):
        state = self.game.get_initial_state()
        moves = [3, 2, 2, 1, 1, 0, 1, 0, 0, 3, 0]
        for m in moves:
            state = self.game.apply_move(state, m)
        is_term, val = self.game.check_terminal(state)
        self.assertTrue(is_term)
        self.assertEqual(val, -1.0)

    def test_draw_game(self):
        state = self.game.get_initial_state()
        draw_sequence = [
            0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0,
            2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
            4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4,
            6, 6, 6, 6, 6, 6
        ]
        for m in draw_sequence:
            is_term, _ = self.game.check_terminal(state)
            if is_term:
                break
            state = self.game.apply_move(state, m)
        is_term, val = self.game.check_terminal(state)
        self.assertTrue(is_term)
        self.assertEqual(val, 0.0)

    def test_mirror_state_and_policy(self):
        state = self.game.get_initial_state()
        state = self.game.apply_move(state, 1)
        mirrored = self.game.mirror_state(state)
        self.assertEqual(mirrored[1, 5, 5], 1.0)

        policy = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.4, 0.0], dtype=np.float32)
        mirrored_policy = self.game.mirror_policy(policy)
        np.testing.assert_array_equal(
            mirrored_policy,
            np.array([0.0, 0.4, 0.0, 0.0, 0.3, 0.2, 0.1], dtype=np.float32)
        )

    def test_board_conversion_roundtrip(self):
        state = self.game.get_initial_state()
        state = self.game.apply_move(state, 3)
        board = self.game.state_to_board(state)
        self.assertEqual(board[5][3], 1)
        reconstructed = self.game.board_to_state(board, current_player=2)
        np.testing.assert_array_equal(state, reconstructed)


if __name__ == "__main__":
    unittest.main()
