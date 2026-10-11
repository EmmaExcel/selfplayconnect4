import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add_game(self, game_history: list):
        for item in game_history:
            self.buffer.append(item)

    def sample(self, batch_size: int) -> tuple:
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        states = np.array([self.buffer[i][0] for i in indices], dtype=np.float32)
        policies = np.array([self.buffer[i][1] for i in indices], dtype=np.float32)
        values = np.array([self.buffer[i][2] for i in indices], dtype=np.float32)
        return states, policies, values

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


def temperature_sample(visit_counts: np.ndarray, temperature: float) -> int:
    if temperature < 1e-8:
        return int(np.argmax(visit_counts))

    counts = visit_counts.astype(np.float64)
    counts = np.power(counts, 1.0 / temperature)
    total = np.sum(counts)
    if total == 0:
        legal = (visit_counts > 0).astype(np.float64)
        if np.sum(legal) == 0:
            return 0
        probs = legal / np.sum(legal)
    else:
        probs = counts / total
    return int(np.random.choice(len(probs), p=probs))


def get_temperature(move_number: int, threshold: int = 15,
                    high: float = 1.0, low: float = 0.1) -> float:
    if move_number < threshold:
        return high
    return low
