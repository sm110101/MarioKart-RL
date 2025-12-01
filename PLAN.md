## Objective
Stand up a clean, importable Python RL project that trains an agent to complete a lap on Yoshi Falls (Mario Kart DS) using the DeSmuME emulator via py-desmume. The project favors low-dimensional state vectors read from emulator memory, discrete action combinations, PPO (RLlib), and an optional Behavior Cloning warm start using recorded demonstrations. The repository should install cleanly, import without py-desmume present, and include tests that skip gracefully when system dependencies or ROMs are missing.

## High-level Milestones
1) Scaffold repo, folders, boilerplate, and configs
2) Implement emulator wrapper stub with safe imports and input stepping
3) Implement reward/memory helpers and state preprocessing utilities
4) Implement Gymnasium environment (state-based)
5) Implement training scripts (PPO, BC) and evaluation script
6) Add notebooks for smoke testing and memory search
7) Add unit tests that skip when deps/ROM absent
8) Write docs, README, .gitignore, LICENSE
9) Sanity-check imports and run smoke tests

## Repository Structure (to create)
- src/ with package-style layout and `__init__.py` files
- src/envs/ `mario_kart_yoshi_falls_env.py`
- src/emulator/ `py_desmume_interface.py`
- src/reward/ `mkds_memory_addresses.py`
- src/training/ `train_ppo.py`, `train_bc.py`, `evaluate_policy.py`
- src/utils/ `state_preprocessing.py`, `viz.py`
- configs/ `env.yaml`, `ppo_default.yaml`
- notebooks/ `00_env_smoke_test.ipynb`, `01_memory_search_mkds.ipynb`
- tests/ `test_env_basic.py`, `test_state_preprocessing.py`
- docs/ `design.md`, `experiments.md`, `results.md`
- Top-level: `.gitignore`, `README.md`, `requirements.txt`, `LICENSE`
- Optional: `data/` (excluded from VCS) for demos/checkpoints

## Dependencies (requirements.txt)
- gymnasium
- numpy
- ray[rllib]
- py-desmume
- matplotlib
- pytest
- pyyaml

Note: On Windows, py-desmume binary availability varies. Guard imports so the package remains importable without py-desmume.

## Emulator Wrapper (src/emulator/py_desmume_interface.py)
Class: `MarioKartInterface`
- Import guarded:
  - try: `from desmume.emulator import DeSmuME`
  - try: `from desmume.controls import Keys, KEYMASK` (names per py-desmume)
  - except: define placeholders and raise helpful runtime errors only on use
- Constructor args:
  ```
  def __init__(
      self,
      rom_path: str,
      savestate_path: str | None = None,
      frames_per_step: int = 4,
      speed_addr: int | None = None,
      lap_addr: int | None = None,
      progress_addr: int | None = None,
      extra_addrs: dict[str, int] | None = None,
  ) -> None
  ```
- Responsibilities:
  - Instantiate emulator
  - Load ROM, optionally savestate (start line Yoshi Falls)
  - Manage stepping and input key presses
  - Read memory via `self.emu.memory.unsigned[...]`
- Methods (docstrings + TODOs):
  - `reset() -> None`: resets, loads savestate if provided, ensures running
  - `step_action(action_id: int) -> None`: discrete ID → keymask; press for `frames_per_step` cycles; release
  - `get_state() -> dict[str, float | int | None]`: reads memory (lap, speed, progress, extras)
  - `get_lap() -> int | None`, `get_speed_raw() -> int | None`, `get_progress_raw() -> int | None`
- Action mapping (initial):
  - 0: no-op
  - 1: A
  - 2: A+LEFT
  - 3: A+RIGHT
  - 4: A+DOWN (brake/drift)
  - 5: release (coast)
- Live viewing: no special render; assume emulator window is visible
- Notes:
  - Add clear TODOs for exact memory addresses and calibration
  - Provide helpful error messages if py-desmume is missing

## Reward and Memory (src/reward/mkds_memory_addresses.py)
- Constants (placeholders):
  - `LAP_ADDR: int | None = None`
  - `SPEED_ADDR: int | None = None`
  - `PROGRESS_ADDR: int | None = None`
  - `MAX_RAW_SPEED: int = 1023` (placeholder)
  - `MAX_RAW_PROGRESS: int = 65535` (placeholder)
- Helpers:
  - `normalize_speed(raw: int | None) -> float` → [0,1], TODO calibrate
  - `normalize_progress(raw: int | None) -> float` → [0,1], TODO calibrate
  - `compute_reward(prev_state: dict, curr_state: dict) -> float`:
    - small time penalty per step (-0.001)
    - positive for normalized speed
    - large bonus for lap increase
    - TODO: more shaping later

## Utils (src/utils/)
state_preprocessing.py
- `state_dict_to_vector(state: dict) -> np.ndarray` (float32)
- Include normalized speed/progress; lap (one-hot or scalar)
- `vector_size() -> int` returns feature count; keep consistent with env
viz.py
- `plot_learning_curve(rewards: list[float], path: str | None = None) -> None`
- `plot_metric(history: dict[str, list[float]], path: str | None = None) -> None`

## Gymnasium Environment (src/envs/mario_kart_yoshi_falls_env.py)
Class: `MarioKartYoshiFallsEnv(gymnasium.Env)`
- Config-driven constructor (dict | None):
  - rom_path, savestate_path, frames_per_step, memory addresses, max_steps_per_episode, extras
  - Instantiate `MarioKartInterface` with config
- Spaces:
  - `action_space = spaces.Discrete(6)` (semantics above)
  - `observation_space = spaces.Box(low=-inf, high=inf, shape=(N,), dtype=float32)` with TODO to update N
- Methods:
  - `reset(seed=None, options=None) -> (np.ndarray, dict)`:
    - emulator.reset()
    - read raw state → `state_dict_to_vector`
    - init counters; return obs, info
  - `step(action: int) -> (np.ndarray, float, bool, bool, dict)`:
    - emulator.step_action(action)
    - read new state
    - reward via `compute_reward`
    - terminated on lap completion (lap increase vs baseline)
    - truncated on max_steps or “stuck” TODO
    - return obs, reward, terminated, truncated, info
  - `metadata = {"render_modes": ["human"]}`
  - `render(mode="human")`: no-op; emulator window provides live view
- Robustness:
  - Skip/raise informative errors if emulator unavailable

## Training (src/training/)
train_ppo.py (RLlib PPO)
- Load `configs/env.yaml` and `configs/ppo_default.yaml` (PyYAML)
- `register_env` for `MarioKartYoshiFallsEnv`
- PPO config: framework="torch", MLP with `[256,256]`, relu
- Optional warm start:
  - If `bc_init_path` provided, TODO: load BC weights into PPO policy
- Build+train loop:
  - `algo = config.build()`
  - For N iters: `algo.train()`, print `episode_reward_mean`
  - TODO: save checkpoints, write metrics (ray_results/)

train_bc.py (Behavior Cloning)
- Dataset: states `(N, obs_dim)` and actions `(N,)` (discrete IDs)
- PyTorch policy net: MLP → logits over actions
- Train with cross-entropy
- Save weights (e.g., `checkpoints/bc_model.pt`)
- Clear TODOs for dataset paths and PPO integration

evaluate_policy.py
- Load PPO checkpoint
- Run K episodes in `MarioKartYoshiFallsEnv`
- Let emulator window display live
- Collect rewards/lap completions; print summary

## Configs (configs/)
env.yaml (template)
- rom_path: "/path/to/mariokartds.nds"
- savestate_path: "/path/to/yoshi_falls_start_state.dsv"
- frames_per_step: 4
- max_steps_per_episode: 2000
- lap_addr: null
- speed_addr: null
- progress_addr: null
- use_timer: false
- extra_addrs: {}

ppo_default.yaml
- framework: "torch"
- gamma: 0.99
- lr: 3e-4
- train_batch_size: 4096
- sgd_minibatch_size: 256
- num_sgd_iter: 10
- model:
  - fcnet_hiddens: [256, 256]
  - fcnet_activation: "relu"
  - vf_share_layers: false
- bc_init_path: null

## Notebooks
00_env_smoke_test.ipynb
- Create env with partial config; reset and random step
- Inspect raw state and processed vector

01_memory_search_mkds.ipynb
- Sample emulator memory over time
- Log candidate addresses; visualize trends to identify lap/speed/progress

## Tests (pytest)
test_env_basic.py
- Instantiate env with dummy config; skip if py-desmume/ROM missing
- reset() returns vector shape/dtype float32; info is dict
- step() returns correct types; finite reward; booleans for terminated/truncated

test_state_preprocessing.py
- state_dict_to_vector handles normal and missing values
- Returns expected shape and dtype

## Docs
docs/design.md
- Problem statement; environment design; discrete actions; reward/termination
- Emulator integration and memory access
- Training approach: BC warm start, PPO fine-tuning

docs/experiments.md
- Reward shaping variants; feature sets; BC vs no-BC; PPO hypers

docs/results.md
- Placeholders for learning curves, lap completion, qualitative notes

## Top-level Files
README.md
- What this is; architecture overview; installation; config instructions
- Recording demonstrations (TODO pointers)
- How to run BC training, PPO (with warm start), evaluation
- Note on ROM/savestate not included

.gitignore
- Python artifacts; ray_results/; checkpoints/; *.pt/*.pth/*.npz
- Media/logs; ROM/savestate extensions: *.nds, *.dsv, *.sav

LICENSE
- MIT license (permissive)

## Implementation Order (Step-by-step)
1) Create folders and empty `__init__.py` files across src subpackages
2) Write `requirements.txt` and `.gitignore`
3) Add configs with templates
4) Implement `state_preprocessing.py` and `viz.py`
5) Implement reward/memory helpers (placeholders)
6) Implement emulator wrapper with guarded imports and action stepping
7) Implement Gymnasium environment (using utils/reward)
8) Implement PPO/BC/evaluate scripts (skeletons, config load)
9) Add notebooks (basic content or empty with markdown headings)
10) Add tests that skip gracefully without deps/ROM
11) Write docs skeletons and README; add LICENSE
12) Local smoke test:
   - `python -c "import src"` (package import check)
   - Run tests with pytest; verify skips occur on missing deps
   - Open `00_env_smoke_test.ipynb` and run random steps (if ROM available)

## Windows/py-desmume Considerations
- If py-desmume fails to install, ensure imports are guarded so the rest of the repo remains usable
- Add clear error messages in emulator wrapper when functionality is invoked without py-desmume
- Prefer absolute ROM/savestate paths in `env.yaml` on Windows

## Acceptance Criteria
- Project installs with `pip -r requirements.txt`
- `import src` succeeds even if py-desmume not installed
- `pytest` runs and skips emulator-dependent tests when deps/ROM missing
- PPO/BC/evaluate scripts parse configs and build stubs without runtime crashes (no training required yet)
- Notebooks open and show placeholders/smoke tests

## Next Steps After Scaffolding
- Identify true memory addresses via `01_memory_search_mkds.ipynb` and field calibration
- Finalize state vector features and update `vector_size()` and env
- Tune reward shaping; implement stuck detection and better termination
- Record demonstrations; implement dataset loader and BC training loop
- Connect BC weights into PPO initialization path and train end-to-end