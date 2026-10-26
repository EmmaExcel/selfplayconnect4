import os
import threading
import torch
import yaml
import numpy as np
from connect4.game import Connect4Game
from connect4.network import Connect4Network
from connect4.mcts import MCTS
from connect4.tactical_search import choose_move


class ModelInference:
    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pt", config_path: str = "configs/default.yaml"):
        self.game = Connect4Game()
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        net_cfg = self.config["network"]
        self.network = Connect4Network(
            num_residual_blocks=net_cfg["num_residual_blocks"],
            num_channels=net_cfg["num_channels"]
        )

        self.checkpoint_path = checkpoint_path
        self.lock = threading.RLock()
        self.last_mtime = 0.0
        self.reload_weights(checkpoint_path)

    def reload_weights(self, checkpoint_path: str = None):
        with self.lock:
            path = checkpoint_path or self.checkpoint_path
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if "model_state_dict" in checkpoint:
                    self.network.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.network.load_state_dict(checkpoint)
                self.last_mtime = os.path.getmtime(path)
            self.network.eval()
            self.network.clear_cache()

    def get_move(self, board: list, current_player: int, simulations: int = 200) -> dict:
        with self.lock:
            return self._get_move(board, current_player, simulations)

    def _get_move(self, board: list, current_player: int, simulations: int) -> dict:
        if os.path.exists(self.checkpoint_path):
            current_mtime = os.path.getmtime(self.checkpoint_path)
            if current_mtime > self.last_mtime:
                self.reload_weights(self.checkpoint_path)

        state = self.game.board_to_state(board, current_player=current_player)
        mcts_cfg = {
            "num_simulations": max(10, simulations),
            "c_puct": self.config["mcts"]["c_puct"]
        }
        if simulations >= 400:
            depth = 10 if simulations >= 1600 else 8 if simulations >= 800 else 6
            action = choose_move(board, current_player, depth=depth)
            policy = np.zeros(self.game.COLS, dtype=np.float32)
            policy[action] = 1.0
        else:
            mcts = MCTS(self.game, self.network, mcts_cfg)
            action, policy = mcts.get_action(state, temperature=0.0, add_noise=False)
        _, value_arr = self.network.predict(state)
        value = float(value_arr[0, 0])

        return {
            "column": int(action),
            "policy": [float(p) for p in policy],
            "value": round(value, 4)
        }
