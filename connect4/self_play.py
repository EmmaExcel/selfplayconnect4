import copy
import numpy as np
import torch
from connect4.game import Connect4Game
from connect4.mcts import MCTS
from connect4.utils import get_temperature


class SelfPlayRunner:
    def __init__(self, game: Connect4Game, mcts_cfg: dict,
                 opponent_pool=None, pool_fraction: float = 0.2, device="cpu"):
        self.game = game
        self.mcts_cfg = mcts_cfg
        self.opponent_pool = opponent_pool
        self.pool_fraction = pool_fraction
        self.device = device

    def generate_games(self, network, num_games: int) -> list:
        all_data = []
        primary_mcts = MCTS(self.game, network, self.mcts_cfg)

        pool_opponent = None
        if self.opponent_pool is not None and self.opponent_pool.size() > 0:
            opponent_state = self.opponent_pool.sample()
            if opponent_state is not None:
                pool_opponent = copy.deepcopy(network)
                pool_opponent.load_state_dict(opponent_state)
                pool_opponent.to(self.device)
                pool_opponent.eval()

        for _ in range(num_games):
            use_pool = (pool_opponent is not None and np.random.random() < self.pool_fraction)
            opponent_mcts = MCTS(self.game, pool_opponent, self.mcts_cfg) if use_pool else primary_mcts

            game_history = self._play_game(primary_mcts, opponent_mcts)
            all_data.extend(game_history)

        return all_data

    def _play_game(self, mcts_current: MCTS, mcts_opponent: MCTS) -> list:
        state = self.game.get_initial_state()
        history = []
        move_number = 0
        current_mcts = mcts_current

        while True:
            is_terminal, _ = self.game.check_terminal(state)
            if is_terminal:
                break

            temperature = get_temperature(
                move_number,
                self.mcts_cfg.get('temperature_threshold_move', 15),
                self.mcts_cfg.get('temperature_high', 1.0),
                self.mcts_cfg.get('temperature_low', 0.1),
            )

            action, policy = current_mcts.get_action(state, temperature=temperature, add_noise=True)
            history.append((state.copy(), policy.copy()))

            state = self.game.apply_move(state, action)
            move_number += 1

            current_mcts = mcts_opponent if current_mcts is mcts_current else mcts_current

        _, terminal_value = self.game.check_terminal(state)

        result = []
        for i, (s, p) in enumerate(history):
            moves_from_end = len(history) - i
            if terminal_value == 0:
                z = 0.0
            elif moves_from_end % 2 == 0:
                z = terminal_value
            else:
                z = -terminal_value

            result.append((s, p, z))
            result.append((self.game.mirror_state(s), self.game.mirror_policy(p), z))

        return result
