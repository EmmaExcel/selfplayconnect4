import argparse
import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from connect4.game import Connect4Game
from connect4.network import Connect4Network
from connect4.mcts import MCTS


def main():
    parser = argparse.ArgumentParser(description="Play Connect 4 against trained agent in terminal")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--human-first", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    game = Connect4Game()
    net_cfg = config["network"]
    network = Connect4Network(
        num_residual_blocks=net_cfg["num_residual_blocks"],
        num_channels=net_cfg["num_channels"]
    )

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        network.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No checkpoint found. Agent will play using initialized network weights.")

    network.eval()
    mcts_config = {
        "num_simulations": args.simulations,
        "c_puct": config["mcts"]["c_puct"]
    }
    mcts = MCTS(game, network, mcts_config)

    state = game.get_initial_state()
    human_turn = args.human_first

    print("Connect 4 Game Started")
    print(f"Human is Player {'1 (X)' if human_turn else '2 (O)'}")
    print(f"AI is Player {'2 (O)' if human_turn else '1 (X)'}")
    game.display(state)

    while True:
        is_terminal, val = game.check_terminal(state)
        if is_terminal:
            if val == 0:
                print("Game ended in a draw.")
            elif human_turn:
                print("AI won the game.")
            else:
                print("Human won the game.")
            break

        legal_moves = game.get_legal_moves(state)
        valid_cols = [c for c in range(game.COLS) if legal_moves[c] == 1]

        if human_turn:
            while True:
                user_input = input(f"Enter column ({', '.join(map(str, valid_cols))}): ")
                try:
                    col = int(user_input.strip())
                    if col in valid_cols:
                        break
                    print("Invalid column. Column is full or out of bounds.")
                except ValueError:
                    print("Please enter a valid integer.")
            state = game.apply_move(state, col)
        else:
            print("AI is calculating move...")
            action, policy = mcts.get_action(state, temperature=0.0, add_noise=False)
            print(f"AI chose column {action}")
            state = game.apply_move(state, action)

        game.display(state)
        human_turn = not human_turn


if __name__ == "__main__":
    main()
