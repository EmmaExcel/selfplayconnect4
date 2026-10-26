import os
import yaml
import torch
import torch.optim as optim
import numpy as np

from connect4.game import Connect4Game
from connect4.network import Connect4Network
from connect4.mcts import MCTS
from connect4.arena import Arena
from connect4.opponent_pool import OpponentPool
from connect4.utils import ReplayBuffer
from connect4.loss import AlphaZeroLoss
from connect4.self_play import SelfPlayRunner


class Trainer:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.game = Connect4Game()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net_cfg = self.config['network']
        self.network = Connect4Network(
            num_residual_blocks=net_cfg['num_residual_blocks'],
            num_channels=net_cfg['num_channels'],
        ).to(self.device)

        checkpoint_path = "checkpoints/best_model.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.network.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        self.network.eval()

        self.best_network = Connect4Network(
            num_residual_blocks=net_cfg['num_residual_blocks'],
            num_channels=net_cfg['num_channels'],
        ).to(self.device)
        self.best_network.load_state_dict(self.network.state_dict())
        self.best_network.eval()

        train_cfg = self.config['training']
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg['num_iterations'] * train_cfg['epochs_per_iteration'],
        )

        self.loss_fn = AlphaZeroLoss()
        self.replay_buffer = ReplayBuffer(max_size=train_cfg['replay_buffer_size'])
        self.opponent_pool = OpponentPool()

        mcts_cfg = dict(self.config['mcts'])
        if 'num_simulations' in train_cfg and 'num_simulations' not in mcts_cfg:
            mcts_cfg['num_simulations'] = train_cfg['num_simulations']

        self.self_play_runner = SelfPlayRunner(
            game=self.game,
            mcts_cfg=mcts_cfg,
            opponent_pool=self.opponent_pool,
            pool_fraction=self.config['opponent_pool'].get('pool_game_fraction', 0.2),
            device=self.device
        )

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("data", exist_ok=True)

    def train(self):
        train_cfg = self.config['training']
        pool_cfg = self.config['opponent_pool']

        for iteration in range(1, train_cfg['num_iterations'] + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{train_cfg['num_iterations']}")
            print(f"{'='*60}")

            print(f"  Self-play: generating {train_cfg['games_per_iteration']} games...")
            game_data = self.self_play_runner.generate_games(
                self.best_network,
                train_cfg['games_per_iteration']
            )
            self.replay_buffer.add_game(game_data)
            print(f"  Replay buffer size: {len(self.replay_buffer)}")

            print(f"  Training for {train_cfg['epochs_per_iteration']} epochs...")
            avg_loss, avg_policy_loss, avg_value_loss = self._train_network(train_cfg)
            print(f"  Loss: {avg_loss:.4f} (policy: {avg_policy_loss:.4f}, value: {avg_value_loss:.4f})")

            print(f"  Arena: evaluating against best network...")
            accepted = self._evaluate()
            if accepted:
                print(f"  New network accepted as best.")
            else:
                print(f"  New network rejected. Keeping previous best.")

            if iteration % pool_cfg['save_interval'] == 0:
                self.opponent_pool.add(self.best_network.state_dict(), iteration)
                print(f"  Saved checkpoint to opponent pool (pool size: {self.opponent_pool.size()})")

            self._save_checkpoint(iteration)

    def _train_network(self, train_cfg: dict) -> tuple:
        self.network.train()
        self.network.clear_cache()
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        for _ in range(train_cfg['epochs_per_iteration']):
            states, target_policies, target_values = self.replay_buffer.sample(train_cfg['batch_size'])

            states_t = torch.from_numpy(states).float().to(self.device)
            target_policies_t = torch.from_numpy(target_policies).float().to(self.device)
            target_values_t = torch.from_numpy(target_values).float().to(self.device).unsqueeze(1)

            legal_moves_mask = (states_t[:, 0, :, :] + states_t[:, 1, :, :])[:, 0, :]
            legal_moves_mask = (legal_moves_mask == 0).float()

            log_policy, value = self.network(states_t, legal_moves_mask)
            loss, policy_loss, value_loss = self.loss_fn(
                log_policy, value, target_policies_t, target_values_t
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss_sum += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            num_batches += 1

        avg_total = total_loss_sum / max(num_batches, 1)
        avg_policy = policy_loss_sum / max(num_batches, 1)
        avg_value = value_loss_sum / max(num_batches, 1)
        self.network.clear_cache()
        return avg_total, avg_policy, avg_value

    def _evaluate(self) -> bool:
        self.network.eval()
        self.best_network.eval()
        eval_mcts_cfg = {
            **self.config['mcts'],
            'num_simulations': min(self.config['mcts'].get('num_simulations', 800), 200)
        }

        new_mcts = MCTS(self.game, self.network, eval_mcts_cfg)
        best_mcts = MCTS(self.game, self.best_network, eval_mcts_cfg)

        def new_player(state):
            action, _ = new_mcts.get_action(state, temperature=0.0, add_noise=False)
            return action

        def best_player(state):
            action, _ = best_mcts.get_action(state, temperature=0.0, add_noise=False)
            return action

        arena = Arena(self.game, new_player, best_player, self.config['arena']['num_games'])
        new_wins, best_wins, draws = arena.play_games()

        total_decisive = new_wins + best_wins
        win_rate = 0.5 if total_decisive == 0 else (new_wins / total_decisive)

        print(f"  Arena results: new={new_wins}, best={best_wins}, draws={draws} (win rate: {win_rate:.2%})")

        threshold = self.config['training']['acceptance_threshold']
        if win_rate >= threshold:
            self.best_network.load_state_dict(self.network.state_dict())
            self.best_network.clear_cache()
            return True

        self.network.load_state_dict(self.best_network.state_dict())
        self.network.clear_cache()
        return False

    def _save_checkpoint(self, iteration: int):
        path = os.path.join("checkpoints", "best_model.pt")
        temp_path = path + ".tmp"
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.best_network.state_dict(),
        }, temp_path)
        os.replace(temp_path, path)
