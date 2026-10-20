import os
import threading
import torch
import torch.optim as optim
import numpy as np

from connect4.game import Connect4Game
from connect4.network import Connect4Network


class OnlineLearner:
    def __init__(self, model_path: str = "checkpoints/best_model.pt", data_path: str = "data/online_buffer.pt"):
        self.model_path = model_path
        self.data_path = data_path
        self.game = Connect4Game()
        self.lock = threading.Lock()
        self.active_sessions = {}
        self.games_learned = 0
        self.model_generation = 1

        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        self.buffer = self._load_buffer()

    def _load_buffer(self) -> list:
        if os.path.exists(self.data_path):
            try:
                data = torch.load(self.data_path, map_location="cpu")
                self.games_learned = data.get("games_learned", 0)
                self.model_generation = data.get("model_generation", 1)
                return data.get("samples", [])
            except Exception:
                return []
        return []

    def _save_buffer(self):
        torch.save({
            "games_learned": self.games_learned,
            "model_generation": self.model_generation,
            "samples": self.buffer[-50000:]
        }, self.data_path)

    def record_move(self, session_id: str, state: np.ndarray, policy: list, player: int):
        with self.lock:
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = []
            policy_arr = np.array(policy, dtype=np.float32)
            self.active_sessions[session_id].append((state.copy(), policy_arr, player))

    def complete_game(self, session_id: str, winner: int, moves: list = None, inference_engine=None):
        with self.lock:
            history = self.active_sessions.pop(session_id, None) or []
            new_samples = []

            if moves:
                state = self.game.get_initial_state()
                current_player = 1
                for col in moves:
                    policy = np.zeros(self.game.COLS, dtype=np.float32)
                    policy[col] = 1.0

                    if winner == 0:
                        z = 0.0
                    elif winner == current_player:
                        z = 1.0
                    else:
                        z = -1.0

                    new_samples.append((state.copy(), policy, z))
                    mirrored_s = self.game.mirror_state(state)
                    mirrored_p = self.game.mirror_policy(policy)
                    new_samples.append((mirrored_s, mirrored_p, z))

                    state = self.game.apply_move(state, col)
                    current_player = 3 - current_player
            elif history:
                for s, p, player in history:
                    if winner == 0:
                        z = 0.0
                    elif winner == player:
                        z = 1.0
                    else:
                        z = -1.0

                    new_samples.append((s, p, z))
                    mirrored_s = self.game.mirror_state(s)
                    mirrored_p = self.game.mirror_policy(p)
                    new_samples.append((mirrored_s, mirrored_p, z))

            if new_samples:
                self.buffer.extend(new_samples)
                self.games_learned += 1
                self._save_buffer()

        if new_samples:
            threading.Thread(
                target=self._run_learning_step,
                args=(inference_engine,),
                daemon=True
            ).start()

        return {
            "games_learned": self.games_learned,
            "model_generation": self.model_generation
        }

    def _run_learning_step(self, inference_engine):
        with self.lock:
            if len(self.buffer) < 16:
                return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            network = Connect4Network().to(device)

            if os.path.exists(self.model_path):
                ckpt = torch.load(self.model_path, map_location=device)
                if "model_state_dict" in ckpt:
                    network.load_state_dict(ckpt["model_state_dict"])
                else:
                    network.load_state_dict(ckpt)

            network.train()
            optimizer = optim.Adam(network.parameters(), lr=0.0003, weight_decay=1e-4)

            batch_size = min(64, len(self.buffer))
            for _ in range(4):
                indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
                states = np.array([self.buffer[i][0] for i in indices], dtype=np.float32)
                policies = np.array([self.buffer[i][1] for i in indices], dtype=np.float32)
                values = np.array([self.buffer[i][2] for i in indices], dtype=np.float32)

                states_t = torch.from_numpy(states).to(device)
                policies_t = torch.from_numpy(policies).to(device)
                values_t = torch.from_numpy(values).unsqueeze(1).to(device)

                mask = (states_t[:, 0, :, :] + states_t[:, 1, :, :])[:, 0, :]
                mask = (mask == 0).float()

                log_p, val = network(states_t, legal_moves_mask=mask)
                safe_log_p = torch.where(policies_t > 0, log_p, torch.zeros_like(log_p))
                loss_p = -torch.mean(torch.sum(policies_t * safe_log_p, dim=1))
                loss_v = torch.mean((val - values_t) ** 2)
                loss = loss_p + loss_v

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model_generation += 1
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(network.state_dict(), self.model_path)
            self._save_buffer()

            if inference_engine is not None:
                inference_engine.reload_weights(self.model_path)

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "games_learned": self.games_learned,
                "model_generation": self.model_generation
            }
