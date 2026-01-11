# A2C (Advantage Actor-Critic)

## Overview ✨

**A2C (Advantage Actor-Critic)** is a synchronous policy-gradient reinforcement learning algorithm that trains an actor (policy) and a critic (value function) together. This repository contains a minimal, readable PyTorch implementation aimed at experimentation and learning. The included example targets OpenAI Gym environments (e.g., `CartPole-v1`).

---

## Features 🔧

- Minimal, well-commented implementation of A2C
- Configurable hyperparameters in `config.py`
- Clear separation: `returns`, `losses`, `networks`, and `train`
- Per-episode metrics returned by `training_loop` for easy inspection

---

## Quick Start — Installation & Run 🚀

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run training:

```bash
python main.py
# or run a short smoke run from REPL
python -c "from train import training_loop; training_loop(1)"
```

Notes:
- The training loop prints progress every 100 episodes by default.
- `training_loop` returns a `metrics` list of dicts (episode, total_reward, loss) for programmatic inspection.

---

## Project Structure 📁

- `main.py` — entry point (calls `training_loop`)
- `train.py` — training loop and core logic
- `returns.py` — `calculate_returns(rewards, next_value, gamma)` utility
- `losses.py` — `compute_loss(log_probs, advantage, values, entropies)`
- `networks.py` — shared actor-critic network(s)
- `config.py` — hyperparameters (GAMMA, N_STEPS, COEF_ENTROPY, VALUE_COEF)
- `requirements.txt` — dependencies
- `README.md` — this file

---

## Configuration ⚙️

Edit `config.py` to change hyperparameters:

```python
COEF_ENTROPY = 9.1
VALUE_COEF = 0.5
N_STEPS = 5
GAMMA = 0.99
```

- `N_STEPS` controls the number of steps collected per update.
- `GAMMA` is the discount factor used in `calculate_returns`.

---

## How it works (brief) 🧠

1. Collect `N_STEPS` of transitions: states, actions, rewards, values, log_probs, entropies.
2. Compute bootstrap `next_value` and discounted returns with `calculate_returns`.
3. Advantage = returns - values.
4. Loss = policy loss + value loss + entropy loss (computed in `compute_loss`).
5. Perform an optimizer step using the aggregated loss.

---

## Outputs & Logging 📝

- Per-episode metrics are appended to a `metrics` list and returned by `training_loop`.
- Progress is printed every 100 episodes by default:

```
Episode: 100 -- Return: <value>  loss: <value>
```

If you prefer another reporting frequency, I can add a `log_interval` parameter to `training_loop`.

---

## Testing & Validation ✅

Recommended tests (not included by default):
- Unit tests for `calculate_returns`
- Unit tests for `compute_loss`
- A smoke test for `training_loop` (1 episode)

To run tests (if added):

```bash
pip install pytest
pytest -q
```

---

## Troubleshooting & Tips ⚠️

- Avoid module-level calls to `training_loop` or heavy functions during import — these can create circular import issues.
- Add `assert returns.shape == values.shape` to catch shape mismatches early.
- If training seems unstable, try reducing `N_STEPS` or lowering the learning rate.

---

## Extending the Project 💡

- Add a configurable `log_interval` parameter to `training_loop`.
- Add checkpoint saving and resume functionality.
- Add unit tests under `tests/` and set up CI (GitHub Actions).
- Add Jupyter notebooks for visualization of training progress.

---

## License & Credits 📜

See the `LICENSE` file for project license details.

---

If you want, I can add tests or a `log_interval` parameter next — tell me which and I’ll implement it. 
