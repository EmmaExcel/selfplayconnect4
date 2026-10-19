import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from connect4.trainer import Trainer


def main():
    cfg = {
        "game": {
            "rows": 6,
            "cols": 7
        },
        "network": {
            "num_residual_blocks": 5,
            "num_channels": 128
        },
        "training": {
            "num_iterations": 10,
            "games_per_iteration": 20,
            "num_simulations": 30,
            "batch_size": 64,
            "epochs_per_iteration": 5,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "replay_buffer_size": 100000,
            "acceptance_threshold": 0.52
        },
        "mcts": {
            "num_simulations": 30,
            "c_puct": 1.25,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "temperature_threshold_move": 10,
            "temperature_high": 1.0,
            "temperature_low": 0.1
        },
        "opponent_pool": {
            "save_interval": 2,
            "pool_game_fraction": 0.25
        },
        "arena": {
            "num_games": 10
        }
    }

    import yaml
    temp_cfg_path = "configs/train_fast.yaml"
    with open(temp_cfg_path, "w") as f:
        yaml.dump(cfg, f)

    trainer = Trainer(config_path=temp_cfg_path)
    trainer.train()

    if os.path.exists(temp_cfg_path):
        os.remove(temp_cfg_path)


if __name__ == "__main__":
    main()
