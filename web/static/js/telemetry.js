export function updateEvaluation(valueDisplay, evalBarFill, value) {
  const normalizedVal = Math.max(-1.0, Math.min(1.0, value));
  valueDisplay.textContent = (normalizedVal > 0 ? "+" : "") + normalizedVal.toFixed(2);

  const centerPercent = 50;
  const offsetPercent = normalizedVal * 50;

  if (normalizedVal >= 0) {
    evalBarFill.style.left = `${centerPercent}%`;
    evalBarFill.style.width = `${offsetPercent}%`;
    evalBarFill.style.backgroundColor = "var(--accent-primary)";
  } else {
    evalBarFill.style.left = `${centerPercent + offsetPercent}%`;
    evalBarFill.style.width = `${Math.abs(offsetPercent)}%`;
    evalBarFill.style.backgroundColor = "var(--disc-red)";
  }
}

export function updatePolicyBars(policy) {
  for (let c = 0; c < 7; c++) {
    const pct = Math.round((policy[c] || 0) * 100);
    const bar = document.getElementById(`policyBar${c}`);
    const label = document.getElementById(`policyPct${c}`);
    if (bar && label) {
      bar.style.width = `${pct}%`;
      label.textContent = `${pct}%`;
    }
  }
}

export function updateModelStats(modelGenDisplay, gamesLearnedDisplay, stats) {
  if (modelGenDisplay && stats.model_generation !== undefined) {
    modelGenDisplay.textContent = stats.model_generation;
  }
  if (gamesLearnedDisplay && stats.games_learned !== undefined) {
    gamesLearnedDisplay.textContent = stats.games_learned;
  }
  const trainingState = document.getElementById("trainingState");
  if (trainingState) {
    trainingState.textContent = stats.training_error
      ? `Training error: ${stats.training_error}`
      : stats.training ? "Learning from completed games…"
      : stats.pending_games ? "Game saved; waiting for enough positions to train…"
      : "Training is up to date";
  }
}

export function renderMoveHistory(moveLog, moveHistory, humanSide) {
  if (moveHistory.length === 0) {
    moveLog.innerHTML = '<div class="move-log-empty">No moves played yet.</div>';
    return;
  }

  moveLog.innerHTML = "";
  moveHistory.slice().reverse().forEach(item => {
    const row = document.createElement("div");
    row.className = "move-log-item";
    const who = item.player === humanSide ? "Human" : "AI";
    const discColor = item.player === 1 ? "Red" : "Yellow";
    row.innerHTML = `<span>#${item.moveNum} ${who} (${discColor})</span><span>Col ${item.col}</span>`;
    moveLog.appendChild(row);
  });
}
