import { ROWS, COLS, getDropRow, checkWin, checkDraw } from "./js/rules.js";
import { fetchMove, fetchStats, reportGameOver } from "./js/api.js";
import { updateEvaluation, updatePolicyBars, updateModelStats, renderMoveHistory } from "./js/telemetry.js";

let board = [];
let humanSide = 1;
let currentTurn = 1;
let isGameOver = false;
let isThinking = false;
let moveCount = 0;
let moveHistory = [];
let currentSessionId = "";

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

  const cell = document.querySelector(`.cell[data-row="${row}"][data-col="${col}"]`);
  if (cell) {
    const disc = cell.querySelector(".cell-disc");
    disc.className = `cell-disc ${player === 1 ? "red" : "yellow"}`;
  }

  moveHistory.push({ moveNum: moveCount, player, col, row });
  renderMoveHistory(moveLog, moveHistory, humanSide);

  const winCells = checkWin(board, player);
  if (winCells) {
    isGameOver = true;
    highlightWin(winCells);
    const winnerName = player === humanSide ? "Human" : "AI Agent";
    statusMessage.textContent = `Game Over. ${winnerName} won the match. Model updating...`;
    handleGameOver(player);
    return;
  }

  if (checkDraw(board)) {
    isGameOver = true;
    statusMessage.textContent = "Game Over. Match ended in a draw. Model updating...";
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
  } catch (err) {}
}

async function triggerAIMove() {
  isThinking = true;
  clearColumnHighlights();
  statusMessage.textContent = "AI Agent is calculating move with MCTS...";
  updateStatusDisplay();

  const simulations = parseInt(difficultySelect.value, 10) || 200;

  try {
    const data = await fetchMove(board, currentTurn, simulations, currentSessionId);
    updateEvaluation(valueDisplay, evalBarFill, data.value);
    updatePolicyBars(data.policy);
    makeMove(data.column, currentTurn);
  } catch (err) {
    statusMessage.textContent = "Error communicating with AI engine.";
  } finally {
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

playerSideButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    playerSideButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    humanSide = parseInt(btn.dataset.side, 10);
    initBoard();
  });
});

initBoard();
