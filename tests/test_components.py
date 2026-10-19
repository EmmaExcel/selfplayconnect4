import unittest
import numpy as np
import torch

from connect4.game import Connect4Game
from connect4.network import Connect4Network
from connect4.node import MCTSNode
from connect4.loss import AlphaZeroLoss
from connect4.self_play import SelfPlayRunner
from connect4.bitboard import pieces_to_bitboard, has_won, get_tactical_moves


class TestComponents(unittest.TestCase):
    def test_bitboard_win_detection(self):
        bb = (1 << 0) | (1 << 7) | (1 << 14) | (1 << 21)
        self.assertTrue(has_won(bb))

        bb_non = (1 << 0) | (1 << 7) | (1 << 14)
        self.assertFalse(has_won(bb_non))

    def test_tactical_moves(self):
        state = np.zeros((3, 6, 7), dtype=np.float32)
        state[0, 5, 0] = state[0, 5, 1] = state[0, 5, 2] = 1.0
        wins, blocks, suicides = get_tactical_moves(state)
        self.assertIn(3, wins)

    def test_node_ucb_score(self):
        state = np.zeros((3, 6, 7), dtype=np.float32)
        node = MCTSNode(state, prior=0.5)
        self.assertEqual(node.q_value, 0.0)

        score = node.ucb_score(sqrt_parent_visits=2.0, c_puct=1.25)
        self.assertGreater(score, 0.0)

    def test_alpha_zero_loss(self):
        loss_fn = AlphaZeroLoss()
        log_p = torch.log(torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True))
        val = torch.tensor([[0.2]], requires_grad=True)

        target_p = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        target_v = torch.tensor([[1.0]])

        total, p_loss, v_loss = loss_fn(log_p, val, target_p, target_v)
        self.assertGreater(total.item(), 0.0)
        total.backward()
        self.assertIsNotNone(val.grad)

    def test_self_play_runner(self):
        game = Connect4Game()
        net = Connect4Network(num_residual_blocks=2, num_channels=16)
        mcts_cfg = {"num_simulations": 5, "c_puct": 1.25}
        runner = SelfPlayRunner(game=game, mcts_cfg=mcts_cfg)
        samples = runner.generate_games(net, num_games=1)
        self.assertGreater(len(samples), 0)
        self.assertEqual(samples[0][0].shape, (3, 6, 7))


if __name__ == "__main__":
    unittest.main()
