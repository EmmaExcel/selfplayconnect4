import unittest
import torch
import numpy as np
from connect4.network import Connect4Network


class TestConnect4Network(unittest.TestCase):
    def setUp(self):
        self.net = Connect4Network(num_residual_blocks=2, num_channels=32)

    def test_forward_output_shapes(self):
        x = torch.zeros((4, 3, 6, 7), dtype=torch.float32)
        policy, value = self.net(x)
        self.assertEqual(policy.shape, (4, 7))
        self.assertEqual(value.shape, (4, 1))

    def test_legal_moves_masking(self):
        x = torch.zeros((1, 3, 6, 7), dtype=torch.float32)
        mask = torch.tensor([[1, 0, 1, 0, 1, 0, 0]], dtype=torch.float32)
        log_policy, _ = self.net(x, legal_moves_mask=mask)
        policy = torch.exp(log_policy).detach().numpy()[0]
        self.assertAlmostEqual(policy[1], 0.0, places=5)
        self.assertAlmostEqual(policy[3], 0.0, places=5)
        self.assertAlmostEqual(policy[5], 0.0, places=5)
        self.assertAlmostEqual(policy[6], 0.0, places=5)
        self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=5)

    def test_backward_pass(self):

if __name__ == '__main__':
    unittest.main()
