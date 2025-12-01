Mario Kart DS RL: Actionable Plan (Image Observations with py-desmume + Gym + CNN)


0) Scope, assumptions, and success criteria

- Goal: Train an agent to drive in Mario Kart DS using pixel observations (84x84 grayscale stacks) with a CNN policy (DQN or PPO), integrating py-desmume as the emulator backend.
- Observation: Stacked images only (no RAM in observation). Reward/termination may use RAM-based signals (hybrid) for learnability.
- Platform: Windows (py-desmume), Python 3.10+, Stable-Baselines3 (SB3), Gymnasium.
- Success (Phase 1): Agent completes 1-lap time trial on Yoshi Falls with meaningful forward progress (>70% lap completion) and average speed > threshold over evaluation episodes.
- Success (Phase 2): Robust lap completion across multiple savestates and basic curriculum tracks; visible improvements in logs/metrics.


1) Milestones, deliverables, and exit criteria

- M1: Emulator wrapper (py-desmume)
  - Deliverables: `src/emulator/nds.py` with APIs: init/load ROM, load savestate, set/clear inputs, run N frames, get RGB frame (top screen), uncapped FPS.
  - Exit: Headless/fast mode verified; single-step loop reaches thousands of FPS-equivalent (no 60 FPS cap), stable frame capture shape.

- M2: Image pipeline and frame stack
  - Deliverables: `src/vision/preprocess.py` with crop(top screen) → grayscale → resize to (84,84) → uint8; `FrameStacker` producing (4,84,84).
  - Exit: One-off validation script saves sample frames before/after preprocessing; visual sanity checked (no bars/rotations).

- M3: Gym environment (`MarioKartDSEnv`)
  - Deliverables: `src/envs/mario_kart_ds_env.py` implementing Gymnasium Env with action space (discrete combos), frame_skip, reset via savestate, step logic, info dict fields.
  - Exit: `env.reset/step` returns correct `obs.shape==(4,84,84)`; deterministic reset from savestate; basic random policy runs ~5k steps without crash.

- M4: Reward & termination (hybrid)
  - Deliverables: RAM readers for progress/speed/off-road/lap (in emulator wrapper), reward function and termination conditions in env, reward clipping wrapper.
  - Exit: Reward correlates with forward progress; stuck detection triggers; episodes terminate at lap completion.

- M5: Training loop (SB3) and configs
  - Deliverables: `src/train/train_dqn.py` (and optional `train_ppo.py`), `src/configs/*.yaml` for hyperparameters; periodic checkpoints, evaluation loop, TensorBoard logging.
  - Exit: 1M steps run completes; learning curves show increasing episode return/progress.

- M6: Debugging & logging
  - Deliverables: Episode video recorder (GIF/MP4) every N steps; metrics: mean ep reward, lap completion rate, average speed, off-road rate; seeds recorded.
  - Exit: Visual playback confirms agent behavior; metrics enable regression comparison.

- M7 (Optional): Behavior cloning warm-start
  - Deliverables: Data logger while playing (obs/action), dataset NPZ; minimal PyTorch BC script; weight transfer into SB3 policy.
  - Exit: RL training from BC-initialized weights converges faster than scratch.

- M8 (Optional): Curriculum and vectorization
  - Deliverables: Multiple savestates for curricula; `DummyVecEnv` with 2–4 envs; track cycling.
  - Exit: Stability/throughput improved; generalization across starts tracks better.


2) Emulator integration plan (py-desmume)

- Headless / fast mode
  - Disable FPS limiting/vsync. If full headless isn’t supported, keep GUI minimized and avoid blocking UI calls.
  - Target: Run loop where `cycle()` advances frames without delay; measure throughput.

- Input semantics
  - Define `apply_buttons(emulator, buttons)` that sets input for a single frame.
  - In `step(action)`: for `frame_skip` frames, set inputs → run frame → optionally clear after each frame. Be consistent so action maps to same held-frames pattern.

- Frame capture
  - Always capture full RGB frame from emulator; crop top screen once; verify shape with a one-off script.
  - Store frames as `np.uint8` for speed; convert to grayscale and resize to 84x84 using `cv2.INTER_AREA`.

- Savestate & determinism
  - Load a savestate at the race start in `reset()`; advance ~10 frames post-load to settle countdown/UI if needed.

- RAM access (for reward/termination)
  - Implement helper reads: progress along track, speed, off-road flag, lap number. Keep previous progress for deltas.

API crib-sheet (py-desmume)

- ROM control:
  - `DeSmuME.open(file_name: str, auto_resume=False)` → load ROM without immediately running.
  - `DeSmuME.reset()`, `DeSmuME.pause()`, `DeSmuME.resume(keep_keypad=False)`.
- Stepping:
  - `DeSmuME.cycle(with_joystick=True)` → advance one frame. Call in a loop for `frame_skip` frames.
- Frame capture:
  - `DeSmuME.screenshot()` → PIL Image (simple, slightly slower).
  - `DeSmuME.display_buffer_as_rgbx(reuse_buffer=True)` → memoryview (fast path). Convert to NumPy and crop top screen:

```python
buf = emu.display_buffer_as_rgbx(reuse_buffer=True)  # memoryview of RGBX
arr = np.frombuffer(buf, dtype=np.uint8).reshape(384, 256, 4)  # H, W, 4
rgb = arr[..., :3]   # drop X
top = rgb[:192]      # top screen (256x192); verify once on your machine
```

- Input:
  - `DeSmuME.input` → access keypad/joystick state via `DeSmuME_Input` (use to press/release buttons consistently per frame).
  - Note: `resume(keep_keypad=False)` resets keypad (releases keys).
- Savestates:
  - `DeSmuME.savestate` → load/save race-start states (use in `reset()`).
- Windowing (optional):
  - `DeSmuME.create_sdl_window(auto_pause=False, use_opengl_if_possible=True)` if you need a GUI; otherwise stay headless/minimized.

Reference: py-desmume API docs for `DeSmuME`, `.input`, `.savestate`, `.screenshot`, `.display_buffer_as_rgbx`, `.cycle()`.


3) Observation pipeline (Atari-style)

- Preprocess: crop top screen → grayscale → resize to (84,84) → uint8; optionally normalize later in policy.
- Stack: maintain last 4 frames via `FrameStacker` to produce `(4,84,84)`.
- Optional: Add a “velocity-ish” difference channel (absdiff vs previous pre-resize gray) as a 5th channel later if needed.
- Consistency: keep dtype uint8 [0,255] unless policy expects float; SB3 CNNs handle uint8 fine.


4) Action space (initial, minimal)

- Start with compact discrete combos to simplify exploration:
  - 0: []
  - 1: ["A"]                    (accelerate)
  - 2: ["A","LEFT"]
  - 3: ["A","RIGHT"]
  - 4: ["LEFT"]
  - 5: ["RIGHT"]
- Expand later (e.g., brake, drift) only if needed for performance.
- Each action is held for `frame_skip` frames per step for stable control.


5) Reward shaping and terminations (hybrid recommended)

- Reward = Δprogress + 0.01*speed − 0.1*off_road_penalty, clipped to [-1,1].
- Terminated when lap completes or no progress for N seconds of game time; `truncated` at `_max_steps` safety cap.
- If RAM lookup is blocked early, temporarily use weaker visual heuristics (e.g., optical flow magnitude) to bootstrap, then switch to RAM-based reward ASAP.


6) Environment API details (Gymnasium)

- `reset(seed)`
  - Load savestate → settle frames → get first processed frame → stack to 4.
  - Return `(obs, info)` with `obs.shape==(4,84,84)`.
- `step(action)`
  - Map action → buttons; for `frame_skip`: set inputs, run frame, (optional) clear.
  - Build next obs via frame stacker; compute reward; set terminated/truncated; return `(obs, reward, terminated, truncated, info)`.
- `info` should include: raw speed, delta_progress, lap, off_road flag, episode frame count.
- Wrap with reward clipping and optional monitor/video wrappers.


7) Training configuration (SB3)

- DQN (baseline):
  - `learning_rate=1e-4`, `buffer_size=100_000`, `batch_size=32`, `learning_starts=10_000`,
    `gamma=0.99`, `train_freq=(4,"step")`, `target_update_interval=10_000`,
    `policy="CnnPolicy"` (custom features extractor optional), `verbose=1`.
- PPO (alternative):
  - `learning_rate=2.5e-4`, `n_steps=128–256`, `batch_size=64–256`, `gamma=0.99`,
    `clip_range=0.1–0.2`, `ent_coef=0.01`, `vf_coef=0.5`, gradient clipping on.
- Vectorization: start with single env; scale to `DummyVecEnv([make_env]*2..4)` after stability check.
- Checkpointing: save model every N env steps; run evaluation episodes saving videos.


8) Behavior cloning warm-start (optional but recommended)

- Dataset: `np.savez_compressed("mkds_bc_dataset.npz", obs=(N,4,84,84), actions=(N,))`.
- Logger: while playing, record action ID and processed frame/stack per emulator step.
- BC training: CNN encoder + linear head, cross-entropy on actions. Transfer encoder weights to SB3 policy before RL.


9) Debugging, logging, and evaluation

- Metrics: mean episode reward, lap completion rate, average speed, off-road rate, time-to-lap.
- Videos: record full evaluation episode every K training steps for qualitative checks.
- Seeds: fix and record seeds for reproducibility; log environment build parameters, ROM path, savestate used, frame_skip, action set.
- Troubleshooting checklist:
  - If learning is flat: verify reward signals (delta progress not stuck at 0), frames not black, correct crop, FPS uncapped.
  - If agent spins: reduce action space, increase frame_skip consistency, try PPO for more stable gradients.
  - If overfitting savestate: add multiple start savestates or curriculum.


10) Risks and mitigations

- Emulator perf bottlenecks: keep GUI minimized/headless; avoid per-frame Python overhead (batch small loops, avoid extra copies); ensure `cv2` ops are vectorized.
- RAM address drift: pin ROM version; document memory addresses; encapsulate reads behind a single module.
- Visual pipeline bugs: early one-off visualization and unit tests on crop/resize.
- Action semantics inconsistency: centralize input application and frame_skip behavior.
- Stability: use reward clipping and gradient clipping; consider PPO if DQN unstable.


11) Proposed repository structure

- `src/emulator/nds.py`            (DeSmuME wrapper: ROM, savestate, input, step, frame, RAM reads)
- `src/vision/preprocess.py`       (crop, grayscale, resize; FrameStacker)
- `src/envs/mario_kart_ds_env.py`  (Gymnasium Env)
- `src/train/train_dqn.py`         (SB3 DQN entrypoint + eval/video)
- `src/train/train_ppo.py`         (optional)
- `src/configs/dqn.yaml`           (hyperparams)
- `src/tools/record_dataset.py`    (BC data logger, optional)
- `src/tools/verify_preproc.py`    (one-off visualization script)


12) Concrete next actions (this week)

- [ ] M1: Implement `nds.py` with: open ROM, load savestate, set/clear inputs, `run_frame()`, `get_rgb_frame()`, uncapped FPS mode.
- [ ] M2: Implement `preprocess.py` and `verify_preproc.py`; save and visually inspect sample frames (raw, cropped top, 84x84).
- [ ] M3: Implement `MarioKartDSEnv` with minimal action space and frame_skip; run random policy for 5k steps stable.
- [ ] M4: Add RAM-based reward & termination; clip rewards; log progress/speed during rollouts.
- [ ] M5: Train DQN for 1M steps; enable TensorBoard; checkpoint and record evaluation videos every 50k steps.
- [ ] (Optional) M7: Build simple BC dataset and pretrain; compare learning curves vs scratch.


Notes and design choices rationale

- Image-only observation keeps the learning problem visually grounded while hybrid reward provides dense signal for tractable training.
- Minimal action space plus consistent `frame_skip` reduces exploration complexity and stabilizes early control.
- Early preprocessing verification avoids “garbage in” issues (black bars, wrong screen, flipped axes).
- Vectorized environments and curricula come after single-env stability to avoid compounding complexity too early.


