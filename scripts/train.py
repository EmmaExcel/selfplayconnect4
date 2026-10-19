import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from connect4.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Connect 4 self-play agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    trainer = Trainer(config_path=args.config)
    trainer.train()


if __name__ == "__main__":
    main()
