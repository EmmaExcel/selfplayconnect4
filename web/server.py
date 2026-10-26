import os
import shutil
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web.inference import ModelInference
from web.online_learner import OnlineLearner

app = FastAPI(title="Self-Play Connect 4", docs_url=None, redoc_url=None)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

data_dir = os.environ.get("CONNECT4_DATA_DIR")
if data_dir:
    os.makedirs(data_dir, exist_ok=True)
    model_path = os.path.join(data_dir, "best_model.pt")
    if not os.path.exists(model_path):
        shutil.copy2("checkpoints/best_model.pt", model_path)
    buffer_path = os.path.join(data_dir, "online_buffer.pt")
else:
    model_path = "checkpoints/best_model.pt"
    buffer_path = "data/online_buffer.pt"

inference_engine = ModelInference(checkpoint_path=model_path)
online_learner = OnlineLearner(model_path=model_path, data_path=buffer_path)


class MoveRequest(BaseModel):
    board: list[list[int]]
    player: int = Field(..., ge=1, le=2)
    simulations: int = Field(default=200, ge=10, le=1600)


class GameOverRequest(BaseModel):
    session_id: str
    winner: int = Field(..., ge=0, le=2)
    moves: list[int] = Field(default_factory=list)


@app.api_route("/", methods=["GET", "HEAD"])
def get_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.api_route("/privacy", methods=["GET", "HEAD"])
def get_privacy():
    return FileResponse(os.path.join(static_dir, "privacy.html"))


@app.api_route("/terms", methods=["GET", "HEAD"])
def get_terms():
    return FileResponse(os.path.join(static_dir, "terms.html"))


@app.api_route("/favicon.ico", methods=["GET", "HEAD"])
def get_favicon():
    return FileResponse(os.path.join(static_dir, "favicon.svg"), media_type="image/svg+xml")


@app.post("/api/move")
def calculate_move(req: MoveRequest):
    if len(req.board) != 6 or any(len(row) != 7 for row in req.board):
        raise HTTPException(status_code=400, detail="Board must be 6 rows by 7 columns")
    if any(cell not in (0, 1, 2) for row in req.board for cell in row):
        raise HTTPException(status_code=400, detail="Board cells must be 0, 1, or 2")
    for col in range(7):
        if any(req.board[row][col] and req.board[row + 1][col] == 0 for row in range(5)):
            raise HTTPException(status_code=400, detail="Board has a floating piece")
    red = sum(cell == 1 for row in req.board for cell in row)
    yellow = sum(cell == 2 for row in req.board for cell in row)
    if not ((red == yellow and req.player == 1) or (red == yellow + 1 and req.player == 2)):
        raise HTTPException(status_code=400, detail="Player does not match board turn")
    state = inference_engine.game.board_to_state(req.board, current_player=req.player)
    if inference_engine.game.check_terminal(state)[0]:
        raise HTTPException(status_code=400, detail="Game is already over")

    result = inference_engine.get_move(
        board=req.board,
        current_player=req.player,
        simulations=req.simulations
    )

    return result


@app.post("/api/game-over")
def report_game_over(req: GameOverRequest):
    try:
        stats = online_learner.complete_game(
            session_id=req.session_id,
            winner=req.winner,
            moves=req.moves,
            inference_engine=inference_engine
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return stats


@app.get("/api/stats")
def get_model_stats():
    return online_learner.get_stats()


@app.post("/api/reset")
def reset_board():
    return {"board": [[0] * 7 for _ in range(6)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=True)
