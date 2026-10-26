export async function fetchMove(board, player, simulations) {
  const response = await fetch("/api/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      board: board,
      player: player,
      simulations: simulations
    })
  });
  if (!response.ok) {
    throw new Error(`Move failed with HTTP ${response.status}`);
  }
  return response.json();
}

export async function fetchStats() {
  const response = await fetch("/api/stats");
  if (!response.ok) {
    throw new Error(`Stats fetch failed with HTTP ${response.status}`);
  }
  return response.json();
}

export async function reportGameOver(sessionId, winner, moves) {
  const response = await fetch("/api/game-over", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      winner: winner,
      moves: moves
    })
  });
  if (!response.ok) {
    throw new Error(`Game over report failed with HTTP ${response.status}`);
  }
  return response.json();
}
