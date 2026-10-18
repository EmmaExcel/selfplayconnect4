import argparse
import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from connect4.network import Connect4Network


def main():
    parser = argparse.ArgumentParser(description="Export model checkpoint for inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="checkpoints/exported_model.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    net_cfg = config["network"]
    model = Connect4Network(
        num_residual_blocks=net_cfg["num_residual_blocks"],
        num_channels=net_cfg["num_channels"]
    )

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Saving model with initialized weights.")

    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Exported model weights to {args.output}")


if __name__ == "__main__":
    main()
