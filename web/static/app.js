import { ROWS, COLS, getDropRow, checkWin, checkDraw } from "./js/rules.js?v=20261026-2";
import { fetchMove, fetchStats, reportGameOver } from "./js/api.js?v=20261026-2";
import { updateEvaluation, updatePolicyBars, updateModelStats, renderMoveHistory } from "./js/telemetry.js?v=20261026-2";

let board = [];
let humanSide = 1;
let currentTurn = 1;
let isGameOver = false;
let isThinking = false;
let moveCount = 0;
let moveHistory = [];
let currentSessionId = "";
let gameEpoch = 0;

const boardGrid = document.getElementById("boardGrid");
const columnIndicators = document.querySelectorAll(".col-indicator");
const statusMessage = document.getElementById("statusMessage");
const currentTurnDisc = document.getElementById("currentTurnDisc");
const moveCounter = document.getElementById("moveCounter");
const difficultySelect = document.getElementById("difficultySelect");
const newGameBtn = document.getElementById("newGameBtn");
const playerSideButtons = document.querySelectorAll(".segment-btn");
const valueDisplay = document.getElementById("valueDisplay");
const evalBarFill = document.getElementById("evalBarFill");
const moveLog = document.getElementById("moveLog");
const modelGenDisplay = document.getElementById("modelGenDisplay");
const gamesLearnedDisplay = document.getElementById("gamesLearnedDisplay");
const resultOverlay = document.getElementById("resultOverlay");
const resultTitle = document.getElementById("resultTitle");
const resultDescription = document.getElementById("resultDescription");
const resultIcon = document.getElementById("resultIcon");
const playAgainBtn = document.getElementById("playAgainBtn");

function generateSessionId() {
  return "sess_" + Date.now().toString(36) + "_" + Math.random().toString(36).substring(2, 7);
}

async function loadStats() {
  try {
    const stats = await fetchStats();
    updateModelStats(modelGenDisplay, gamesLearnedDisplay, stats);
  } catch (err) {}
}

function initBoard() {
  gameEpoch++;
  resultOverlay.hidden = true;
  board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
  isGameOver = false;
  isThinking = false;
  moveCount = 0;
  currentTurn = 1;
  moveHistory = [];
  currentSessionId = generateSessionId();

  renderEmptyGrid();
  updateStatusDisplay();
  updateEvaluation(valueDisplay, evalBarFill, 0.0);
  updatePolicyBars([0, 0, 0, 0, 0, 0, 0]);
  renderMoveHistory(moveLog, moveHistory, humanSide);
  loadStats();

  if (humanSide === 2) {
    triggerAIMove();
  }
}

function renderEmptyGrid() {
  boardGrid.innerHTML = "";
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.row = r;
      cell.dataset.col = c;

      const disc = document.createElement("div");
      disc.className = "cell-disc";
      cell.appendChild(disc);

      cell.addEventListener("mouseenter", () => highlightColumn(c));
      cell.addEventListener("mouseleave", () => clearColumnHighlights());
      cell.addEventListener("click", () => handleColumnClick(c));

      boardGrid.appendChild(cell);
    }
  }
}

function highlightColumn(col) {
  if (isGameOver || isThinking) return;
  columnIndicators.forEach((ind, c) => {
    ind.classList.toggle("active", c === col);
  });
}

function clearColumnHighlights() {
  columnIndicators.forEach(ind => ind.classList.remove("active"));
}

function handleColumnClick(col) {
  if (isGameOver || isThinking) return;
  if (currentTurn !== humanSide) return;

  const dropRow = getDropRow(board, col);
  if (dropRow === -1) return;

  makeMove(col, humanSide);

  if (!isGameOver && currentTurn !== humanSide) {
    triggerAIMove();
  }
}

function makeMove(col, player) {
  const row = getDropRow(board, col);
  if (row === -1) return;

  board[row][col] = player;
  moveCount++;
  moveCounter.textContent = `Move: ${moveCount}`;

  const cell = document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
  if (cell) {
    const disc = cell.querySelector(".cell-disc");
    disc.className = `cell-disc dropping ${player === 1 ? "red" : "yellow"}`;
  }

  moveHistory.push({ moveNum: moveCount, player, col, row });
  renderMoveHistory(moveLog, moveHistory, humanSide);

  const winCells = checkWin(board, player);
  if (winCells) {
    isGameOver = true;
    highlightWin(winCells);
    const humanWon = player === humanSide;
    statusMessage.textContent = humanWon ? "You won!" : "AI won this round.";
    showResult(humanWon ? "You won!" : "AI wins!", humanWon ? "Excellent play. Ready for another round?" : "The AI found four in a row. Try again!", humanWon ? "★" : "✦");
    handleGameOver(player);
    return;
  }

  if (checkDraw(board)) {
    isGameOver = true;
    statusMessage.textContent = "It's a draw!";
    showResult("It's a draw!", "Neither side connected four. Play again?", "=");
    handleGameOver(0);
    return;
  }

  currentTurn = player === 1 ? 2 : 1;
  updateStatusDisplay();
}

async function handleGameOver(winner) {
  try {
    const cols = moveHistory.map(m => m.col);
    const stats = await reportGameOver(currentSessionId, winner, cols);
    updateModelStats(modelGenDisplay, gamesLearnedDisplay, stats);
  } catch (err) {
    statusMessage.textContent += " Game could not be saved for training.";
  }
}

function showResult(title, description, icon) {
  resultTitle.textContent = title;
  resultDescription.textContent = description;
  resultIcon.textContent = icon;
  resultOverlay.hidden = false;
  playAgainBtn.focus();
}

async function triggerAIMove() {
  const epoch = gameEpoch;
  isThinking = true;
  clearColumnHighlights();
  statusMessage.textContent = "AI Agent is calculating move with MCTS...";
  updateStatusDisplay();

  const simulations = parseInt(difficultySelect.value, 10) || 150;

  try {
    const data = await fetchMove(board, currentTurn, simulations);
    if (epoch !== gameEpoch) return;
    updateEvaluation(valueDisplay, evalBarFill, data.value);
    updatePolicyBars(data.policy);
    makeMove(data.column, currentTurn);
  } catch (err) {
    if (epoch !== gameEpoch) return;
    statusMessage.textContent = "Error communicating with AI engine.";
  } finally {
    if (epoch !== gameEpoch) return;
    isThinking = false;
    if (!isGameOver) {
      updateStatusDisplay();
    }
  }
}

function updateStatusDisplay() {
  moveCounter.textContent = `Move: ${moveCount}`;
  currentTurnDisc.className = `disc-preview ${currentTurn === 1 ? "" : "yellow"}`;

  if (isGameOver) return;

  if (isThinking) {
    statusMessage.textContent = "AI Agent is calculating move with MCTS...";
  } else if (currentTurn === humanSide) {
    const colorName = humanSide === 1 ? "Red" : "Yellow";
    statusMessage.textContent = `Your turn (${colorName}). Click any column to drop a piece.`;
  } else {
    statusMessage.textContent = "AI turn. Waiting for agent...";
  }
}

function highlightWin(cells) {
  cells.forEach(([r, c]) => {
    const cell = document.querySelector(`.cell[data-row="${r}"][data-col="${c}"]`);
    if (cell) {
      const disc = cell.querySelector(".cell-disc");
      disc.classList.add("winning");
    }
  });
}

newGameBtn.addEventListener("click", () => {
  initBoard();
});

playAgainBtn.addEventListener("click", initBoard);
resultOverlay.addEventListener("click", event => {
  if (event.target === resultOverlay) initBoard();
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !resultOverlay.hidden) initBoard();
});

playerSideButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    playerSideButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    humanSide = parseInt(btn.dataset.side, 10);
    initBoard();
  });
});

initBoard();
setInterval(loadStats, 2500);
