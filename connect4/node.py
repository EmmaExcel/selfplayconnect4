class MCTSNode:
    __slots__ = ['state', 'parent', 'action', 'children', 'visit_count',
                 'value_sum', 'prior', 'is_terminal', 'terminal_value', 'is_expanded']

    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_terminal = False
        self.terminal_value = None
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, sqrt_parent_visits: float, c_puct: float) -> float:
        q = -self.q_value
        u = c_puct * self.prior * sqrt_parent_visits / (1 + self.visit_count)
        return q + u
