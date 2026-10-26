import os
import json
import sqlite3
import threading
import torch
import torch.optim as optim
import numpy as np

from connect4.game import Connect4Game
from connect4.network import Connect4Network


class OnlineLearner:
    """Persist finished games, then train one background worker on their positions."""

    def __init__(self, model_path: str = "checkpoints/best_model.pt", data_path: str = "data/online.sqlite3"):
        self.model_path = model_path
        self.data_path = data_path if data_path.endswith(".sqlite3") else os.path.splitext(data_path)[0] + ".sqlite3"
        self.legacy_path = (data_path if data_path.endswith(".pt") else
                            os.path.join(os.path.dirname(self.data_path), "online_buffer.pt"))
        self.game = Connect4Game()
        self.lock = threading.Lock()
        self.games_learned = 0
        self.model_generation = 0
        self.completed_sessions = set()
        self.training = False
        self.pending_games = 0
        self.training_error = None
        os.makedirs(os.path.dirname(self.data_path) or ".", exist_ok=True)
        self.db = sqlite3.connect(self.data_path, check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS games (
                session_id TEXT PRIMARY KEY, winner INTEGER NOT NULL,
                moves TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state BLOB NOT NULL, policy BLOB NOT NULL, value REAL NOT NULL
            );
        """)
        self.buffer = self._load_buffer()
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                self.model_generation = max(self.model_generation, int(checkpoint.get("online_generation", 0)))
        if self.pending_games and len(self.buffer) >= 16:
            self.training = True
            threading.Thread(target=self._learning_worker, args=(None,), daemon=True).start()

    def _load_buffer(self) -> list:
        # Import the previous torch buffer once; leave the file as a backup.
        if not self.db.execute("SELECT 1 FROM metadata LIMIT 1").fetchone() and os.path.exists(self.legacy_path):
            data = torch.load(self.legacy_path, map_location="cpu", weights_only=False)
            with self.db:
                self.db.executemany("INSERT INTO samples(state, policy, value) VALUES (?, ?, ?)",
                                    (self._encode_sample(sample) for sample in data.get("samples", [])))
                self.db.executemany("INSERT OR IGNORE INTO games(session_id, winner, moves) VALUES (?, 0, '[]')",
                                    ((session_id,) for session_id in data.get("completed_sessions", [])))
                for key in ("games_learned", "model_generation", "pending_games"):
                    self.db.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", (key, int(data.get(key, 0))))
        values = dict(self.db.execute("SELECT key, value FROM metadata"))
        self.games_learned = values.get("games_learned", 0)
        self.model_generation = values.get("model_generation", 0)
        self.pending_games = values.get("pending_games", 0)
        self.completed_sessions = {row[0] for row in self.db.execute("SELECT session_id FROM games")}
        return [self._decode_sample(row) for row in self.db.execute(
            "SELECT state, policy, value FROM samples ORDER BY id DESC LIMIT 50000"
        ).fetchall()[::-1]]

    @staticmethod
    def _encode_sample(sample):
        state, policy, value = sample
        return (sqlite3.Binary(np.asarray(state, dtype=np.float32).tobytes()),
                sqlite3.Binary(np.asarray(policy, dtype=np.float32).tobytes()), float(value))

    @staticmethod
    def _decode_sample(row):
        return (np.frombuffer(row[0], dtype=np.float32).reshape(3, 6, 7).copy(),
                np.frombuffer(row[1], dtype=np.float32).copy(), float(row[2]))

    @staticmethod
    def _atomic_save(payload, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        temp_path = path + ".tmp"
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _save_buffer(self):
        with self.db:
            self.db.executemany("INSERT INTO metadata(key, value) VALUES (?, ?) "
                                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (
                ("games_learned", self.games_learned),
                ("model_generation", self.model_generation),
                ("pending_games", self.pending_games),
            ))

    def _replay(self, moves: list[int], winner: int) -> list:
        if not moves or len(moves) > 42:
            raise ValueError("A finished game needs 1 to 42 moves")
        state = self.game.get_initial_state()
        player = 1
        samples = []
        for index, col in enumerate(moves):
            if not isinstance(col, int) or not 0 <= col < 7:
                raise ValueError("Invalid column")
            if self.game.check_terminal(state)[0]:
                raise ValueError("Moves continue after game over")
            policy = np.zeros(7, dtype=np.float32)
            policy[col] = 1.0
            z = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
            samples.append((state.copy(), policy, z))
            samples.append((self.game.mirror_state(state), self.game.mirror_policy(policy), z))
            state = self.game.apply_move(state, col)
            player = 3 - player
        terminal, value = self.game.check_terminal(state)
        actual_winner = 0 if value == 0 else (3 - player if value == -1 else player)
        if not terminal or winner != actual_winner:
            raise ValueError("Reported result does not match the moves")
        return samples

    def complete_game(self, session_id: str, winner: int, moves: list = None, inference_engine=None):
        # Validate before mutating a session or a persisted counter.
        samples = self._replay(moves or [], winner)
        with self.lock:
            if session_id in self.completed_sessions:
                return self._stats_unlocked()
            with self.db:
                self.db.execute("INSERT INTO games(session_id, winner, moves) VALUES (?, ?, ?)",
                                (session_id, winner, json.dumps(moves)))
                self.db.executemany("INSERT INTO samples(state, policy, value) VALUES (?, ?, ?)",
                                    (self._encode_sample(sample) for sample in samples))
                self.db.execute("DELETE FROM samples WHERE id <= (SELECT COALESCE(MAX(id), 0) - 50000 FROM samples)")
                self.db.execute("INSERT INTO metadata(key, value) VALUES ('games_learned', ?) "
                                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (self.games_learned + 1,))
                self.db.execute("INSERT INTO metadata(key, value) VALUES ('pending_games', ?) "
                                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (self.pending_games + 1,))
            self.buffer = (self.buffer + samples)[-50000:]
            self.games_learned += 1
            self.pending_games += 1
            self.completed_sessions.add(session_id)
            if not self.training:
                self.training = True
                threading.Thread(target=self._learning_worker, args=(inference_engine,), daemon=True).start()
            return self._stats_unlocked()

    def _learning_worker(self, inference_engine):
        while True:
            with self.lock:
                if not self.pending_games or len(self.buffer) < 16:
                    self.training = False
                    return
                queued_games = self.pending_games
                self.pending_games = 0
                samples = list(self.buffer[-50000:])
            try:
                self._run_learning_step(samples, inference_engine)
                with self.lock:
                    self.training_error = None
            except Exception as exc:
                with self.lock:
                    self.pending_games += queued_games
                    self.training_error = str(exc)
                    self.training = False
                return

    def _run_learning_step(self, samples, inference_engine):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = Connect4Network().to(device)
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, map_location=device, weights_only=False)
            network.load_state_dict(ckpt.get("model_state_dict", ckpt))
        network.train()
        optimizer = optim.Adam(network.parameters(), lr=0.0005, weight_decay=1e-4)
        batch_size = min(64, len(samples))
        for _ in range(8):
            indices = np.random.choice(len(samples), size=batch_size, replace=False)
            states = torch.from_numpy(np.stack([samples[i][0] for i in indices])).to(device)
            policies = torch.from_numpy(np.stack([samples[i][1] for i in indices])).to(device)
            values = torch.tensor([samples[i][2] for i in indices], dtype=torch.float32, device=device).unsqueeze(1)
            legal = ((states[:, 0, 0, :] + states[:, 1, 0, :]) == 0).float()
            log_p, val = network(states, legal_moves_mask=legal)
            # A losing human move is evidence about the value of its position,
            # but should not become a policy target for the agent to imitate.
            policy_weight = (values.squeeze(1) >= 0).float()
            per_position_policy_loss = -(policies * torch.where(policies > 0, log_p, 0)).sum(dim=1)
            loss_p = (per_position_policy_loss * policy_weight).sum() / policy_weight.sum().clamp_min(1)
            loss = loss_p + ((val - values) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        network.eval()
        with self.lock:
            next_generation = self.model_generation + 1
            self._atomic_save({"model_state_dict": network.cpu().state_dict(),
                               "online_generation": next_generation}, self.model_path)
            if inference_engine is not None:
                inference_engine.reload_weights(self.model_path)
            self.model_generation = next_generation
            self._save_buffer()

    def _stats_unlocked(self):
        return {
            "games_learned": self.games_learned,
            "model_generation": self.model_generation,
            "training": self.training,
            "pending_games": self.pending_games,
            "training_error": self.training_error,
        }

    def get_stats(self) -> dict:
        with self.lock:
            return self._stats_unlocked()

    def close(self):
        with self.lock:
            self.db.close()
