import os
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

inference_engine = ModelInference()
online_learner = OnlineLearner()


class MoveRequest(BaseModel):
    board: list[list[int]]
    player: int = Field(..., ge=1, le=2)
    simulations: int = Field(default=200, ge=10, le=1600)
    session_id: str | None = None


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

    result = inference_engine.get_move(
        board=req.board,
        current_player=req.player,
        simulations=req.simulations
    )

    if req.session_id:
        state = inference_engine.game.board_to_state(req.board, current_player=req.player)
        online_learner.record_move(req.session_id, state, result["policy"], req.player)

    return result


@app.post("/api/game-over")
def report_game_over(req: GameOverRequest):
    stats = online_learner.complete_game(
        session_id=req.session_id,
        winner=req.winner,
        moves=req.moves,
        inference_engine=inference_engine
    )
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
