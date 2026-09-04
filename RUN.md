# How to run

```bash
pip install -r requirements.txt
```

Everything is a module: run from the project root with `python3 -m src.<thing>`.
Every run writes to `runs/<name>/`.

---

## The 30-second version

```bash
python3 -m src.train --preset smoke          # ~20 min, checks the pipeline works
python3 -m src.viz.dashboard --port 8080     # watch it: http://127.0.0.1:8080
```

## The real thing

```bash
# 1. TRAIN   paper config, 10x10x10, ~2 h on an RTX 4070
python3 -m src.train --preset paper --dataset CUT-2 --run bpp1_cut2

# 2. WATCH   (run in a second terminal, works while training)
python3 -m src.viz.dashboard --port 8080

# 3. TEST    BPP-1, BPP-k MCTS and the baselines on held-out sequences
python3 -m src.evaluate --run bpp1_cut2 --datasets RS CUT-1 CUT-2 \
       --episodes 500 --bppk 2 3 5

# 4. SEE     (step 3 must run first - it records the traces these read)
python3 -m src.viz.replay3d --run bpp1_cut2 --all    # 3D packing GIFs
python3 -m src.viz.heatmap  --run bpp1_cut2          # masks + action probs
python3 -m src.viz.report   --run bpp1_cut2          # -> runs/bpp1_cut2/report.html
```

Or all of it, unattended and resumable:

```bash
./scripts/run_all.sh        # ~12 h: 3 benchmarks + re-orientation + ablation
```

## Training every option the game can offer

```bash
./scripts/train_all_options.sh          # ~5-8 h from scratch, resumable
```

**BPP-1, BPP-3 and BPP-5 are not separate trainings.** BPP-k reuses the BPP-1
network unchanged and only searches harder at test time (paper Sec. 3.3), so the
algorithm is an `--bppk` flag on `src.evaluate`, not a run. The training matrix
is therefore just **3 streams x 2 orientation settings = 6 networks**:

| run | stream | orientations |
|---|---|---|
| `bpp1_cut2` / `bpp1_orient_cut2` | CUT-2 | 1 / 2 |
| `bpp1_cut1` / `bpp1_orient_cut1` | CUT-1 | 1 / 2 |
| `bpp1_rs`   / `bpp1_orient_rs`   | RS    | 1 / 2 |

Each one by hand is:

```bash
python3 -m src.train --preset paper --dataset CUT-2 --orientations 2 \
        --run bpp1_orient_cut2 --total-steps 30000000
python3 -m src.evaluate --run bpp1_orient_cut2 --datasets CUT-2 \
        --episodes 500 --bppk 3 5 --bppk-episodes 100      # all three algorithms
```

Measured on an RTX 4070 Laptop at 30M steps: **1.0-2.0 h** per 1-orientation run,
**1.14x that** for 2 orientations (the action space doubles), and **~9 min** for
the evaluation sweep of one run - BPP-5 dominates it at ~3.8 s/episode against
BPP-1's 0.07 s. Running several trainings at once buys little: three concurrent
jobs dropped each from ~8,300 to ~3,400 steps/s, so they contend rather than
scale. Useful knobs: `STEPS=` to shorten, `ORIENT_ONLY=1` for just the 2-pose
runs, `NO_EVAL=1` to train now and evaluate later.

---

## Making it faster

Profiled at the default `num_envs=32` on an RTX 4070 Laptop + 16 cores: 66% of a
PPO iteration is `collect` (44% the numpy env loop on **one** core, 22% the
per-step GPU round-trip) and 34% is the update. Three knobs, all measured
end-to-end on a trained net:

```bash
# 1.65x - the whole tuned config in one word (num_envs=128 + 40k sequence pool)
python3 -m src.train --preset fast --dataset CUT-2 --run bpp1_cut2_fast

# or the pieces, on top of any preset
python3 -m src.train --preset paper --num-envs 128        # 1.57x  <- almost all of it
python3 -m src.train --preset paper --seq-pool 40000      # 1.05x
python3 -m src.train --preset paper --epochs 2 --minibatches 4   # 1.25x
```

| config | steps/s | vs default | h per 100M |
|---|---|---|---|
| `paper` (32 envs, 4x8) | 4,385 | 1.00x | 6.3 |
| `+ --seq-pool 40000` | 4,613 | 1.05x | 6.0 |
| `fast` (128 envs + pool) | 7,257 | **1.65x** | **3.8** |
| `fast --num-envs 512` | 7,878 | 1.80x | 3.5 |
| `fast --epochs 2 --minibatches 4` | 8,701 | 1.98x | 3.2 |

**`--num-envs` is where the speed is**, because a bigger batch amortises the GPU
round-trip (one forward costs 0.70 ms at batch 32 and 1.17 ms at batch 512) and
the update. The env loop itself does **not** parallelise - it is a serial numpy
loop over the envs, so `env.step` scales linearly with `num_envs`.

> **These are not free.** `--num-envs` and `--epochs`/`--minibatches` both change
> the optimisation: 128 envs takes the batch from 1,280 to 5,120, i.e. 4x fewer
> gradient steps per sample. Higher steps/s is not automatically higher
> utilisation per hour. A/B them on utilisation-vs-**step** over ~5M steps before
> committing a long run, and keep `--preset paper` for reproduction numbers.
> `--seq-pool` is the only one that cannot change a gradient - though a finite
> pool is reused (40,000 sequences is ~1,400 repeats over 100M steps), so do not
> shrink it much.

Running several trainings at once does **not** make one run faster - each process
still uses one core for its env loop - but it does use otherwise idle cores when
there is a queue: 2 concurrent runs = 1.74x aggregate, 3 = 2.18x.

---

# Play against it - human vs. agent

```bash
python3 -m src.viz.game --port 8090          # -> http://127.0.0.1:8090
```

Open that URL in a browser. Safe to run while training is going: it only reads
checkpoints. No checkpoint is needed at all unless you pick a *trained*
opponent - the three heuristics play on a bare 10x10x10 bin.

### 1. Set up the duel

| Choice | What it does |
|---|---|
| **Box stream** | `RS` / `CUT-1` / `CUT-2` - which generator produces the sequence. CUT-1 and CUT-2 always admit a 100% packing; RS usually does not. |
| **Opponent** | who replays your sequence afterwards (table below). Only agents **trained on the stream you picked** are offered, because a network is off-distribution on any other one. Tick *show every agent* to reach the rest; the heuristics belong to every stream and are always listed. |
| **Rotation** | whether boxes may be turned 90 degrees. Only offered against the heuristics, and it applies to *both* players. A trained opponent is locked to the `orientations` its checkpoint was trained with, and the box says which. |
| **Seed** | leave blank for a random stream, or type a number to play an exact one again. |

**You always get exactly the lookahead the opponent gets** - it is not a setting.
BPP-1 and the heuristics see only the box being placed, so *Coming next* is
empty. BPP-k sees k boxes, so you get the other k-1 - *plus* the one box after
the window that it peeks at. That peeked box is drawn dashed and marked
**peek only**: the search never places or reorders it, it only feeds it to the
critic to value the resulting bin, so neither of you may act on it. In box
counts, BPP-3 knows 4 shapes and BPP-5 knows 6, and so do you. The bin size likewise comes
from the chosen opponent's `config.json` when it is a trained one (so the network
gets the geometry it was trained on), otherwise 10x10x10.

### 2. Pack the stream

| Action | How |
|---|---|
| Drop the box | click a cell - the box's front-left-bottom corner lands there |
| See where it lands | hover: the footprint is outlined green (legal) or red, with the resting height and, when illegal, the reason |
| See every legal spot | the *show every legal position* checkbox (blue dots) |
| Turn the box 90 degrees | `R` or the button - only when rotation is on, and never for a square footprint (it has no second pose) |
| Stop early | *I'm stuck - stop here* |
| Abandon the game | *Back to setup*, in the Controls card or the header - it asks first if you have boxes down |
| Turn the 3D bin around | drag it; double-click puts the camera back. On the result screen both bins share one camera, so they always compare at the same angle |

The 3D bin is a free camera - drag to spin it about the vertical axis and tilt
between a near-top view and a near-side elevation (useful for reading stack
heights). It carries **labelled x / y / z axes with unit ticks and the origin
marked `0`**, so you can read any box's position straight off the picture. An
axis fades out where boxes stand in front of it and returns to full strength
where it clears them, so you can see at a glance what is behind what - rotate if
a stretch you care about is buried.

The box you are placing is drawn **to scale**: a 4x2x3 box is four pallet cells
by two, with its height marked on it, so you can hold it against the free space
on the grid. *Coming next* uses the same scale at 60%, so those boxes are
honestly sized against each other too.

Illegal clicks are refused, not punished - the game only ends when no legal
position is left for the arriving box, when the stream runs out, or when you
stop. The stability rule refusing your unsupported placements is exactly the one
the agent is held to.

### 3. Compare

*Let the agent try* hands the agent your exact sequence and scores both
packings: utilisation, boxes packed, why each side stopped, and a slider that
walks the two 3D bins forward box for box so you can see where you diverged.
Then *Rematch* for new boxes, or *Replay this exact sequence* to try the same
stream again.

### Opponents

| Opponent | Needs a run? | Speed | Notes |
|---|---|---|---|
| `BPP-1 agent (<run>)` | yes | ~0.2 s / game | the trained network, greedy, 1 box of lookahead - the paper's main method. It packs from its *predicted* mask, so it can occasionally pick an illegal spot and end its own game - that is the deployed behaviour, not a bug |
| `BPP-3 / BPP-5 MCTS (<run>)` | yes | ~1-10 s / game | same network + permutation tree search over 3 or 5 boxes, plus 1 peeked at for the leaf value (`cfg.mcts_last_item`); you are shown all of them |
| `boundary rule` | no | ~10 s / game | the paper's spare-cuboid heuristic - slow but strong |
| `deepest-bottom-left` | no | instant | classic online rule - the one to warm up against |
| `random feasible` | no | instant | the floor |

Trained opponents are listed automatically for every `runs/<name>/` that has a
`config.json` and a `best.pt` or `latest.pt`, grouped by the `dataset` in that
config. Picking a stream with no matching agent leaves only the heuristics and
says so; ticking *show every agent* adds a "trained on another stream" group and
warns, under the dropdown, that the network is off-distribution. That gap is
real - an agent trained on CUT-1 scores far below its own numbers on RS.

### Flags

| Flag | Default | Why |
|---|---|---|
| `--port` | `8090` | change if 8090 is taken |
| `--host` | `127.0.0.1` | `--host 0.0.0.0` to play from another machine on the network |
| `--device` | `cuda` | `--device cpu` if the GPU is busy training - falls back on its own if there is no CUDA |
| `--mcts-sims` | `100` | simulations per move for the BPP-k opponents: higher = stronger and slower |

Every finished duel is appended to `runs/duel_log.jsonl` and the last few are
shown on the setup screen.

---

> What each dashboard plot means and what a healthy trend looks like:
> **[DASHBOARD.md](DASHBOARD.md)**

## Common flags

| Want to | Command |
|---|---|
| Resume an interrupted run | `python3 -m src.train --run bpp1_cut2 --resume` |
| Stop after N hours | `--max-hours 8` |
| Train on another benchmark | `--dataset RS` or `--dataset CUT-1` |
| Item re-orientation (2 poses) | `--preset orient` |
| Run on CPU | `--device cpu` |
| Ablate the paper's scheme | `--no-mp` / `--no-mc` / `--no-fe` |
| Step through an episode by hand | `python3 -m src.viz.heatmap --run bpp1_cut2 --live` (arrow keys) |
| MP4 instead of GIF | `python3 -m src.viz.replay3d --run bpp1_cut2 --all --mp4` |
| Bigger MCTS search | `python3 -m src.evaluate --run bpp1_cut2 --bppk 5 --mcts-sims 400` |
| Play a duel against the agent | `python3 -m src.viz.game --port 8090` |
| Duel the agent on CPU | `python3 -m src.viz.game --device cpu` |

## Where things land

```
runs/<name>/metrics.jsonl   one line per update - what the dashboard reads
runs/<name>/latest.pt       checkpoint (--resume reads this)
runs/<name>/best.pt         best-utilisation checkpoint (evaluate uses this)
runs/<name>/eval.json       test results
runs/<name>/report.html     self-contained report, open it in a browser
runs/<name>/replays/        GIFs, MP4s, PNGs
runs/duel_log.jsonl         one line per human-vs-agent game
```

## What's running right now

```bash
ps -eo pid,etime,%cpu,args --no-headers | grep src.train | grep -v grep
tail -1 runs/bpp1_cut2/metrics.jsonl | python3 -m json.tool
```

A training run and `scripts/chain_after.sh` are already going in the background;
they survive closing this terminal. Kill them with `pkill -f src.train`.

## If something breaks

- **`FileNotFoundError: traces.json`** - run `src.evaluate` before `replay3d` / `heatmap`.
- **`--live` window doesn't open** - no display; drop `--live` and it writes a GIF.
- **Dashboard is empty** - no `runs/*/metrics.jsonl` yet; start a training run first.
- **CUDA out of memory** - lower `--num-envs` (default 32); for the game, `--device cpu`.
- **The duel has no trained opponents** - no `runs/*/best.pt` yet; train first, or
  play the heuristics, which need no checkpoint.
- **"finish your packing first"** - the agent only runs once your game has ended;
  press *I'm stuck - stop here* if you want to hand it over early.
- **Port already in use** - `python3 -m src.viz.game --port 8091`.
