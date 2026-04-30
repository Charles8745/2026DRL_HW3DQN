# HW3-1: Naive DQN for static mode

> 此 README 為 placeholder，內容待補。
> 完整作業說明請參考 [`report.md`](report.md)。

## Quick Start

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
python -m src.dqn_naive --mode static --epochs 1000 --seed 42
python -m src.dqn_replay --mode static --epochs 1000 --seed 42
python -m src.dqn_replay --mode random --epochs 5000 --seed 42
python -m src.animate --exp naive_static
python -m src.animate --exp replay_static
python -m src.animate --exp replay_random
```
