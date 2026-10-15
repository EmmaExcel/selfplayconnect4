import unittest
import numpy as np
from connect4.game import Connect4Game
from connect4.network import Connect4Network
from connect4.mcts import MCTS


class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.game = Connect4Game()
        self.net = Connect4Network(num_residual_blocks=2, num_channels=32)
        self.config = {
            "num_simulations": 60,
            "c_puct": 1.25,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25
        }
        self.mcts = MCTS(self.game, self.net, self.config)

    def test_search_output_shape_and_legality(self):
        state = self.game.get_initial_state()
        counts = self.mcts.search(state, add_noise=False)
        self.assertEqual(len(counts), 7)
        self.assertEqual(int(np.sum(counts)), 60)

    def test_get_action(self):
        state = self.game.get_initial_state()
        action, policy = self.mcts.get_action(state, temperature=0.0, add_noise=False)
        self.assertIn(action, range(7))
        self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=5)

    def test_mcts_takes_immediate_win(self):
        state = self.game.get_initial_state()
        state = self.game.apply_move(state, 0)
        state = self.game.apply_move(state, 1)
        state = self.game.apply_move(state, 0)
        state = self.game.apply_move(state, 1)
        state = self.game.apply_move(state, 0)
        state = self.game.apply_move(state, 1)

        counts = self.mcts.search(state, add_noise=False)
        chosen_move = int(np.argmax(counts))
        self.assertEqual(chosen_move, 0)


if __name__ == "__main__":
    unittest.main()
