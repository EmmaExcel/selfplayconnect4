# Connect 4 AlphaZero

A lightweight, dependency-minimal implementation of AlphaZero for Connect 4 built with PyTorch, NumPy, and FastAPI.

The agent starts with zero domain knowledge and learns purely through self-play reinforcement learning guided by Monte Carlo Tree Search (MCTS), combined with bitboard tactical evaluation and real-time continuous learning from human matches.

## Core Design

### 1. State Representation
Connect 4 is played on a 6-row by 7-column board. States are represented as `(3, 6, 7)` float32 tensors:
- Plane 0: Current player's pieces (binary mask).
- Plane 1: Opponent's pieces (binary mask).
- Plane 2: Turn indicator (all ones if Player 2, all zeros if Player 1).

Every move inverts the perspective so that the network always evaluates positions from the viewpoint of the player whose turn it is. This halves the effective state space the network needs to explore.

### 2. Dual-Head ResNet
The model uses a shared convolutional backbone with residual connections followed by two task-specific heads:
- Backbone: 5 residual blocks (128 channels each) with batch normalization and ReLU activations.
- Policy Head: Projects spatial features to a 7-dimensional probability vector over columns, with logits of illegal moves masked to `-1e4` prior to log-softmax.
- Value Head: Compresses spatial features through a 256-unit hidden layer with tanh activation, producing a position score in `[-1.0, +1.0]`.

### 3. Bitboard Tactical Acceleration
Game rules and win evaluations are accelerated using 64-bit integer bitboards (`connect4/bitboard.py`). Each column occupies 7 bits (6 rows plus 1 sentinel bit):
- Horizontal check: bit shift by 7.
- Diagonal (\) check: bit shift by 6.
- Diagonal (/) check: bit shift by 8.
- Vertical check: bit shift by 1.

Win detection runs in constant time without nested loops. The tactical engine detects immediate 4-in-a-row wins, mandatory defensive blocks, and suicide moves (moves that hand an opponent a winning drop directly above).

### 4. Search and Training Loop
- MCTS Selection: Upper Confidence Bound for Trees (PUCT) balances exploitation and exploration:
  `Score(s, a) = -Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))`
- Training Loss:
  `L = (v - z)^2 - sum(pi * log(p)) + c * ||theta||^2`
  Where `z` is the final game outcome (`+1.0`, `-1.0`, or `0.0`), `v` is the predicted value, `pi` is the search visit distribution, and `p` is the network policy prior.
- Stability: Historical checkpoints are stored in an opponent pool. 20% of self-play games are played against random past versions to prevent cyclical forgetting.
- Gating: Candidate weights must beat the incumbent champion with at least a 55% win rate over 40 games in an alternating-order arena before being accepted.

### 5. Continuous Online Learning
When hosted as a web service, every completed human game is captured by `web/online_learner.py`. The full move trajectory is labeled, horizontally mirrored, stored in an experience buffer, and trained on via background gradient updates. Updated weights are hot-reloaded by the inference worker without restarting the server.

## Repository Structure

```
selfplayconnect4/
├── connect4/
│   ├── __init__.py          # Public library exports
│   ├── bitboard.py          # 64-bit integer board logic and tactical analysis
│   ├── game.py              # Connect 4 environment and canonical transformations
│   ├── network.py           # Dual-head ResNet (policy and value heads)
│   ├── node.py              # MCTS tree node with UCB scoring
│   ├── mcts.py              # Monte Carlo Tree Search loop
│   ├── loss.py              # AlphaZero cross-entropy and MSE loss module
│   ├── self_play.py         # Self-play game simulation worker
│   ├── arena.py             # Head-to-head match evaluation between checkpoints
│   ├── opponent_pool.py     # Historical model pool manager
│   ├── trainer.py           # Training orchestrator
│   └── utils.py             # Replay buffer and temperature scheduling
├── configs/
│   ├── default.yaml         # Full production hyperparameters
│   └── test_smoke.yaml      # Fast verification configuration
├── scripts/
│   ├── train.py             # Full training entry point
│   ├── train_fast.py        # Fast multi-iteration training script
│   ├── play_cli.py          # Terminal interactive play against the agent
│   └── export_model.py      # Checkpoint export utility
├── web/
│   ├── server.py            # FastAPI service endpoints
│   ├── inference.py         # Model loader with automatic file timestamp reload
│   ├── online_learner.py    # Continuous background learner for human games
│   └── static/
│       ├── index.html       # Game interface with telemetry
│       ├── privacy.html     # Privacy policy
│       ├── terms.html       # Terms and conditions
│       ├── style.css        # Clean dark CSS design system
│       ├── favicon.svg      # Vector favicon
│       ├── app.js           # Controller wiring
│       └── js/
│           ├── api.js       # Client API endpoints
│           ├── rules.js     # Client board mechanics
│           └── telemetry.js # Telemetry charts and evaluation bar
├── tests/
│   ├── test_game.py         # Rules, legality, terminal checks, mirroring
│   ├── test_network.py      # ResNet shapes, masking, gradient flow
│   ├── test_mcts.py         # Search simulations, tactical winning moves
│   ├── test_pipeline.py     # Buffer, pool, temperature, online learner
│   └── test_components.py   # Unit tests for bitboard, node, loss, self-play
├── checkpoints/             # Trained model weights (.pt)
├── pyproject.toml           # Packaging metadata
└── requirements.txt         # Pinned runtime dependencies
```

## Getting Started

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0 or higher

### Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd selfplayconnect4
pip install -r requirements.txt
```

Or install in editable development mode:

```bash
pip install -e .
```

### Running Unit Tests

The test suite contains 27 unit tests verifying the bitboard engine, neural network gradients, MCTS exploration, opponent pool, and online learner:

```bash
python3 -m unittest discover tests
```

### Playing in the Terminal

Play against the current champion checkpoint:

```bash
python3 scripts/play_cli.py --simulations 400
```

Add `--human-first` to play as Player 1 (Red):

```bash
python3 scripts/play_cli.py --human-first --simulations 400
```

### Starting Training

To run the standard training loop with default hyperparameters:

```bash
python3 scripts/train.py --config configs/default.yaml
```

To run a fast multi-iteration training cycle:

```bash
python3 scripts/train_fast.py
```

Checkpoints are automatically written to `checkpoints/best_model.pt` and `checkpoints/pool/`.

### Running the Web Interface

Start the local web server:

```bash
python3 -m uvicorn web.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in any browser:
- Main Game: `http://localhost:8000/`
- Privacy Policy: `http://localhost:8000/privacy`
- Terms and Conditions: `http://localhost:8000/terms`

The web application shows live position value evaluations, MCTS prior probabilities per column, move history, and dynamic model generation statistics.

## License

MIT License.
