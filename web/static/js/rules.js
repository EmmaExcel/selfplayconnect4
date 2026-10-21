export const ROWS = 6;
export const COLS = 7;

export function getDropRow(board, col) {
  for (let r = ROWS - 1; r >= 0; r--) {
    if (board[r][col] === 0) {
      return r;
    }
  }
  return -1;
}

export function checkWin(board, player) {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c <= COLS - 4; c++) {
      if (board[r][c] === player && board[r][c + 1] === player &&
          board[r][c + 2] === player && board[r][c + 3] === player) {
        return [[r, c], [r, c + 1], [r, c + 2], [r, c + 3]];
      }
    }
  }

  for (let r = 0; r <= ROWS - 4; r++) {
    for (let c = 0; c < COLS; c++) {
      if (board[r][c] === player && board[r + 1][c] === player &&
          board[r + 2][c] === player && board[r + 3][c] === player) {
        return [[r, c], [r + 1, c], [r + 2, c], [r + 3, c]];
      }
    }
  }

  for (let r = 0; r <= ROWS - 4; r++) {
    for (let c = 0; c <= COLS - 4; c++) {
      if (board[r][c] === player && board[r + 1][c + 1] === player &&
          board[r + 2][c + 2] === player && board[r + 3][c + 3] === player) {
        return [[r, c], [r + 1, c + 1], [r + 2, c + 2], [r + 3, c + 3]];
      }
    }
  }

  for (let r = 3; r < ROWS; r++) {
    for (let c = 0; c <= COLS - 4; c++) {
      if (board[r][c] === player && board[r - 1][c + 1] === player &&
          board[r - 2][c + 2] === player && board[r - 3][c + 3] === player) {
        return [[r, c], [r - 1, c + 1], [r - 2, c + 2], [r - 3, c + 3]];
      }
    }
  }

  return null;
}

export function checkDraw(board) {
  return board[0].every(cell => cell !== 0);
}
