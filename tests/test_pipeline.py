import unittest
import shutil
import tempfile
import time
import numpy as np
from connect4.game import Connect4Game
from connect4.network import Connect4Network
from connect4.mcts import MCTS
from connect4.arena import Arena
from connect4.opponent_pool import OpponentPool
from connect4.utils import ReplayBuffer, temperature_sample, get_temperature


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.game = Connect4Game()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_replay_buffer(self):
        buf = ReplayBuffer(max_size=10)
        self.assertEqual(len(buf), 0)

        s = np.zeros((3, 6, 7), dtype=np.float32)
        p = np.ones(7, dtype=np.float32) / 7.0
        v = 1.0

        buf.add_game([(s, p, v) for _ in range(5)])
        self.assertEqual(len(buf), 5)

        states, policies, values = buf.sample(3)
        self.assertEqual(states.shape, (3, 3, 6, 7))
        self.assertEqual(policies.shape, (3, 7))
        self.assertEqual(values.shape, (3,))

        buf.clear()
        self.assertEqual(len(buf), 0)

    def test_temperature_utilities(self):
        self.assertEqual(get_temperature(5, threshold=15, high=1.0, low=0.1), 1.0)
        self.assertEqual(get_temperature(20, threshold=15, high=1.0, low=0.1), 0.1)

        counts = np.array([0, 10, 0, 0, 0, 0, 0], dtype=np.float32)
        self.assertEqual(temperature_sample(counts, 0.0), 1)

    def test_opponent_pool(self):
        pool = OpponentPool(pool_dir=self.temp_dir)
        self.assertEqual(pool.size(), 0)
        self.assertIsNone(pool.sample())
        self.assertIsNone(pool.latest())

        net = Connect4Network(num_residual_blocks=2, num_channels=16)
        pool.add(net.state_dict(), 1)
        self.assertEqual(pool.size(), 1)
        self.assertIsNotNone(pool.sample())
        self.assertIsNotNone(pool.latest())

    def test_arena_play(self):
        def first_col_player(state):
            legal = self.game.get_legal_moves(state)
            for c in range(7):
                if legal[c] == 1:
                    return c
            return 0

        def last_col_player(state):
            legal = self.game.get_legal_moves(state)
            for c in range(6, -1, -1):
                if legal[c] == 1:
                    return c
            return 0

        arena = Arena(self.game, first_col_player, last_col_player, num_games=2)
        p1_wins, p2_wins, draws = arena.play_games()
        self.assertEqual(p1_wins + p2_wins + draws, 2)

    def test_online_learner(self):
        from web.online_learner import OnlineLearner
        import os

        model_file = os.path.join(self.temp_dir, "test_model.pt")
        data_file = os.path.join(self.temp_dir, "test_buffer.pt")

        learner = OnlineLearner(model_path=model_file, data_path=data_file)
        self.assertEqual(learner.games_learned, 0)

        moves = [3, 2, 3, 2, 3, 2, 3]
        stats = learner.complete_game("sess_test", winner=1, moves=moves)
        self.assertEqual(stats["games_learned"], 1)
        self.assertGreater(len(learner.buffer), 0)
        self.assertEqual(learner.complete_game("sess_test", winner=1, moves=moves)["games_learned"], 1)
        reloaded = OnlineLearner(model_path=model_file, data_path=data_file)
        self.assertEqual(reloaded.get_stats()["games_learned"], 1)
        self.assertEqual(len(reloaded.buffer), len(learner.buffer))
        with self.assertRaises(ValueError):
            reloaded.complete_game("bad", winner=2, moves=moves)
        self.assertEqual(reloaded.get_stats()["games_learned"], 1)

    def test_online_training_saves_generation(self):
        from web.online_learner import OnlineLearner
        import os
        model_file = os.path.join(self.temp_dir, "trained_model.pt")
        data_file = os.path.join(self.temp_dir, "trained_buffer.pt")
        learner = OnlineLearner(model_path=model_file, data_path=data_file)
        learner.complete_game("finished", winner=2, moves=[3, 2, 3, 2, 4, 2, 4, 2])
        deadline = time.monotonic() + 30
        while learner.get_stats()["training"] and time.monotonic() < deadline:
            time.sleep(0.05)
        stats = learner.get_stats()
        self.assertIsNone(stats["training_error"])
        self.assertEqual(stats["model_generation"], 1)
        self.assertTrue(os.path.exists(model_file))
        self.assertEqual(OnlineLearner(model_file, data_file).get_stats()["model_generation"], 1)


if __name__ == "__main__":
    unittest.main()
