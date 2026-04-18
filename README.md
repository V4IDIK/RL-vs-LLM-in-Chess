# RL vs LLM Chess

A visual chess project where a reinforcement learning (RL) agent plays against a language-model-based (LLM) agent in a Pygame interface.

## What this project does

- Runs a full chess game between two AI players:
  - `RLAgent` (PPO from Stable-Baselines3)
  - `LLMAgent` (GPT-2 based move generator + legal-move filtering/fallback logic)
- Shows the match in a custom GUI with:
  - board rendering
  - move history
  - captured piece info
  - game timer
  - start screen with side selection and optional random seed

## Project structure

- `main.py` - entry point; initializes GUI, loads/trains RL model, and runs the game loop.
- `chess_environment/chess_env.py` - Gym-compatible chess environment and board encoding.
- `rl_player/rl_agent.py` - PPO training/inference wrapper for RL moves.
- `llm_player/llm_agent.py` - GPT-2 move suggestion, parsing, validation, caching, and fallback strategy.
- `chess_gui.py` - Pygame-based UI and game visualization.
- `requirements.txt` - Python dependencies.

## Requirements

- Python 3.9+ recommended
- `pip`
- A machine that can run:
  - Pygame window rendering
  - Hugging Face `gpt2-large` model inference

## Installation

1. Clone the repository and enter the folder:

```bash
git clone <your-repo-url>
cd rl-vs-llm-chess-main
```

2. (Recommended) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Assets required for GUI

The GUI code expects image assets in these locations:

- `icon.png`
- `chess_pieces/wp.png`, `chess_pieces/wn.png`, `chess_pieces/wb.png`, `chess_pieces/wr.png`, `chess_pieces/wq.png`, `chess_pieces/wk.png`
- `chess_pieces/bp.png`, `chess_pieces/bn.png`, `chess_pieces/bb.png`, `chess_pieces/br.png`, `chess_pieces/bq.png`, `chess_pieces/bk.png`

If these files are missing, startup will fail when loading images.

## Running the project

```bash
python main.py
```

At startup:

1. The start screen appears.
2. Choose who plays White (`RL` or `LLM`).
3. Optionally enter a numeric seed (for reproducible randomness).
4. The game begins after RL model load/train.

## RL model behavior

- Default model file: `rl_model.zip`
- Checkpoint file during training: `rl_model_checkpoint.zip`
- If `rl_model.zip` exists, it is loaded.
- Otherwise, the RL agent trains for `50,000` timesteps and saves the model.

You can adjust training behavior in:

- `main.py` (`total_timesteps` / `checkpoint_freq` call site)
- `rl_player/rl_agent.py` (PPO hyperparameters)

## LLM behavior

- Uses Hugging Face `gpt2-large` via `transformers`.
- Prompts with current FEN + legal move list.
- Extracts UCI moves with regex and validates legality.
- Uses fallback heuristics if generation fails (prefers checks/captures, tries to avoid repetition).

## Notes and caveats

- First run may take time because:
  - `gpt2-large` may need to be downloaded
  - RL model may train if no local model exists
- Gameplay speed includes intentional visualization delays in `main.py`.
- The Gym API in this code uses the classic return signature (`obs, reward, done, info`).

## Troubleshooting

- **Pygame window does not open**: ensure local desktop session and graphics support are available.
- **Model download errors**: check internet access and local cache permissions.
- **Slow performance**:
  - expect slower LLM turns on CPU
  - reduce RL training timesteps in `main.py`
- **Missing file errors for images**: add required assets under `icon.png` and `chess_pieces/`.

## Quick start recap

```bash
pip install -r requirements.txt
python main.py
```

Select White side in the GUI and watch RL vs LLM play.
