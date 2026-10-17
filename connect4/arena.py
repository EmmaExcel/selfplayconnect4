import numpy as np
from connect4.game import Connect4Game


class Arena:
    def __init__(self, game: Connect4Game, player1_fn, player2_fn, num_games: int = 40):
        self.game = game
        self.player1_fn = player1_fn
        self.player2_fn = player2_fn
        self.num_games = num_games

    def play_games(self) -> tuple:
        p1_wins = 0
        p2_wins = 0
        draws = 0

        for i in range(self.num_games):
            if i < self.num_games // 2:
                first_fn = self.player1_fn
                second_fn = self.player2_fn
                first_is_p1 = True
            else:
                first_fn = self.player2_fn
                second_fn = self.player1_fn
                first_is_p1 = False

            result = self._play_single_game(first_fn, second_fn)

            if result == 0:
                draws += 1
            elif (result == 1 and first_is_p1) or (result == -1 and not first_is_p1):
                p1_wins += 1
            else:
                p2_wins += 1

        return p1_wins, p2_wins, draws

    def _play_single_game(self, first_fn, second_fn) -> int:
        state = self.game.get_initial_state()
        current_fn = first_fn
        move_count = 0

        while True:
            is_terminal, value = self.game.check_terminal(state)
            if is_terminal:
                if value == 0:
                    return 0
                if move_count % 2 == 0:
                    return int(value)
                return int(-value)

            action = current_fn(state)
            state = self.game.apply_move(state, action)
            move_count += 1

            if current_fn is first_fn:
                current_fn = second_fn
            else:
                current_fn = first_fn
