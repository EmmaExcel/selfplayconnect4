import math
import numpy as np
from connect4.game import Connect4Game
from connect4.node import MCTSNode


class MCTS:
    def __init__(self, game: Connect4Game, network, config: dict):
        self.game = game
        self.network = network
        self.num_simulations = config.get('num_simulations', 800)
        self.c_puct = config.get('c_puct', 1.25)
        self.dirichlet_alpha = config.get('dirichlet_alpha', 0.3)
        self.dirichlet_epsilon = config.get('dirichlet_epsilon', 0.25)

    def search(self, state: np.ndarray, add_noise: bool = True, early_stop: bool = False) -> np.ndarray:
        root = MCTSNode(state)
        self._expand(root)

        if add_noise:
            self._add_dirichlet_noise(root)

        for sim in range(self.num_simulations):
            node = root

            while node.is_expanded and not node.is_terminal:
                node = self._select_child(node)

            if node.is_terminal:
                value = node.terminal_value
            else:
                value = self._expand(node)

            self._backup(node, value)

            if early_stop and sim >= 25 and sim % 4 == 0:
                counts = [child.visit_count for child in root.children.values()]
                if len(counts) > 1:
                    counts.sort(reverse=True)
                    remaining = self.num_simulations - sim - 1
                    if counts[0] > counts[1] + remaining:
                        break

        visit_counts = np.zeros(self.game.COLS, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts

    def _expand(self, node: MCTSNode) -> float:
        is_terminal, terminal_value = self.game.check_terminal(node.state)
        if is_terminal:
            node.is_terminal = True
            node.terminal_value = terminal_value
            node.is_expanded = True
            return terminal_value

        legal_moves = self.game.get_legal_moves(node.state)
        policy, value = self.network.predict(node.state, legal_moves)
        policy = policy.flatten()

        policy = policy * legal_moves
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = legal_moves / np.sum(legal_moves)

        for col in range(self.game.COLS):
            if legal_moves[col] == 1:
                child_state = self.game.apply_move(node.state, col)
                child = MCTSNode(child_state, parent=node, action=col, prior=policy[col])
                is_term, term_val = self.game.check_terminal(child_state)
                if is_term:
                    child.is_terminal = True
                    child.terminal_value = term_val
                    child.is_expanded = True
                node.children[col] = child

        node.is_expanded = True
        return float(value.flatten()[0])

    def _evaluate(self, node: MCTSNode) -> float:
        _, value = self.network.predict(node.state)
        return float(value.flatten()[0])

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        sqrt_parent = math.sqrt(node.visit_count)
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_score(sqrt_parent, self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backup(self, node: MCTSNode, value: float):
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            value = -value
            current = current.parent

    def _add_dirichlet_noise(self, node: MCTSNode):
        if not node.children:
            return

        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))

        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = ((1 - self.dirichlet_epsilon) * child.prior
                           + self.dirichlet_epsilon * noise[i])

    def get_action(self, state: np.ndarray, temperature: float = 0.0,
                   add_noise: bool = False) -> tuple:
        winning_moves, blocking_moves, suicide_moves = self.game.get_tactical_moves(state)

        if winning_moves:
            action = winning_moves[0]
            policy = np.zeros(self.game.COLS, dtype=np.float32)
            policy[action] = 1.0
            return action, policy

        if len(blocking_moves) == 1 and not add_noise:
            action = blocking_moves[0]
            policy = np.zeros(self.game.COLS, dtype=np.float32)
            policy[action] = 1.0
            return action, policy

        legal = self.game.get_legal_moves(state)
        valid_cols = [c for c in range(self.game.COLS) if legal[c] == 1]
        if len(valid_cols) == 1:
            action = valid_cols[0]
            policy = np.zeros(self.game.COLS, dtype=np.float32)
            policy[action] = 1.0
            return action, policy

        visit_counts = self.search(state, add_noise=add_noise, early_stop=(temperature < 1e-8 and not add_noise))

        if blocking_moves and not add_noise:
            for c in range(self.game.COLS):
                if c not in blocking_moves:
                    visit_counts[c] = 0.0

        if suicide_moves and not add_noise:
            legal = self.game.get_legal_moves(state)
            safe_legal = [c for c in range(self.game.COLS) if legal[c] == 1 and c not in suicide_moves]
            if safe_legal:
                for c in suicide_moves:
                    visit_counts[c] = 0.0

        if temperature < 1e-8:
            action = int(np.argmax(visit_counts))
        else:
            counts = visit_counts ** (1.0 / temperature)
            total = np.sum(counts)
            if total == 0:
                legal = self.game.get_legal_moves(state)
                probs = legal / np.sum(legal)
                action = int(np.random.choice(len(probs), p=probs))
            else:
                probs = counts / total
                action = int(np.random.choice(len(probs), p=probs))

        total_visits = np.sum(visit_counts)
        if total_visits > 0:
            policy = visit_counts / total_visits
        else:
            policy = np.zeros(self.game.COLS, dtype=np.float32)

        return action, policy
