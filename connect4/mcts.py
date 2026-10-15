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

    def search(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        root = MCTSNode(state)
        self._expand(root)

        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root

            while node.is_expanded and not node.is_terminal:
                node = self._select_child(node)

            if node.is_terminal:
                value = node.terminal_value
            else:
                self._expand(node)
                if node.is_terminal:
                    value = node.terminal_value
                else:
                    value = self._evaluate(node)

            self._backup(node, value)

        visit_counts = np.zeros(self.game.COLS, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts

    def _expand(self, node: MCTSNode):
        is_terminal, terminal_value = self.game.check_terminal(node.state)
        if is_terminal:
            node.is_terminal = True
            node.terminal_value = terminal_value
            node.is_expanded = True
            return

        legal_moves = self.game.get_legal_moves(node.state)
        policy, _ = self.network.predict(node.state, legal_moves)
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
