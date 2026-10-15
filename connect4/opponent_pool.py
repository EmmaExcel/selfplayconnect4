import os
import torch
import random


class OpponentPool:
    def __init__(self, pool_dir: str = "checkpoints/pool"):
        self.pool_dir = pool_dir
        os.makedirs(pool_dir, exist_ok=True)

    def add(self, model_state_dict: dict, iteration: int):
        path = os.path.join(self.pool_dir, f"checkpoint_{iteration:04d}.pt")
        torch.save(model_state_dict, path)

    def sample(self) -> dict | None:
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            return None
        path = random.choice(checkpoints)
        return torch.load(path, map_location="cpu", weights_only=True)

    def latest(self) -> dict | None:
        checkpoints = self._list_checkpoints()
        if not checkpoints:
