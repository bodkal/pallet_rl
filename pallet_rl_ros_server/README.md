# pallet_rl_ros_server

A ROS 2 (Humble) service that tells the robot **which box to pick next and
where to put it**, using the trained pallet RL policy.

It is **stateless**: every request carries the whole pallet (the boxes already
on it) and the conveyor (the boxes at the station). The server rebuilds the
pallet from them, decides, answers, and forgets. If the robot places a box
somewhere other than where it was told, just send what is really there next time.

It lives inside the training repo (`pallet_rl/pallet_rl_ros_server`) but stands
alone: it imports nothing from `ar2l`, so the folder can be copied to the robot
PC on its own.

## Layout

```
src/pallet_rl_interfaces/          messages (ament_cmake)
    msg/Box.msg                    one carton, real cm, pallet frame
    srv/PlanPlacement.srv          placed[] + station[] -> pick + placement
src/pallet_rl_server/              the node (ament_python)
    config/server.yaml             EVERY setting: service, model, grid, rules, robot
    models/colom_center_top.pt     the policy (weights + training args)
    launch/server.launch.py
    pallet_rl_server/
        node.py                    ROS wrapper, nothing else
        planner.py                 stateless planner: cm in, cm out (no ROS)
        snapshot.py                writes each step to a JSON file for the game
        settings.py                reads server.yaml, checks it against the model
        core/pallet.py             the pallet state + every feasibility rule
        core/rules.py              the rules resolved to grid cells
        core/sweeps.py             sliding-window max helpers
        core/arm.py                UR20 IK + capsule collision check
        core/nets.py               the networks, inference only
    test/test_parity.py            proves it decides exactly like the training code
```

## Setup (once)

ROS Humble uses the system Python 3.10. Torch lives in a venv that can also
see the ROS packages:

```bash
cd ~/Projects/pallet_rl/pallet_rl_ros_server
python3.10 -m venv --without-pip --system-site-packages .venv
curl -sO https://bootstrap.pypa.io/get-pip.py && .venv/bin/python get-pip.py
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/python -m pip install "numpy<2" pyyaml pytest
```

## Build and run

```bash
cd ~/Projects/pallet_rl/pallet_rl_ros_server
source /opt/ros/humble/setup.bash
source .venv/bin/activate          # before colcon AND before running
python -m colcon build
source install/setup.bash

ros2 launch pallet_rl_server server.launch.py
# or with another settings file:
ros2 launch pallet_rl_server server.launch.py config:=/path/to/server.yaml
```

The venv must be active when the node starts (its script runs `python3`
from PATH).

## Calling it

Frame: origin at the pallet corner where the grid starts, x along the pallet
length (124 cm = 31 cells), y along its width (108 cm = 27 cells), z up from
the deck. All values are **real cm**. Types: **0 blue, 1 white, 2 brown**
(the CSV files number them 1/2/3).

```bash
ros2 service call /pallet_rl/plan_placement pallet_rl_interfaces/srv/PlanPlacement "{
  pallet_id: 'demo',
  placed:  [{x_cm: 27, y_cm: 17, z_cm: 0, length_cm: 50, width_cm: 30, height_cm: 28, type: 0}],
  station: [{length_cm: 50,   width_cm: 30,   height_cm: 28, type: 0},
            {length_cm: 45.5, width_cm: 27.5, height_cm: 25, type: 1},
            {length_cm: 30,   width_cm: 20,   height_cm: 20, type: 2}]}"
```

Request:

| field | meaning |
|---|---|
| `placed[]` | every box on the pallet: centre x, y, bottom z, sides **as it stands** (length along x), type. Placement order if you have it; any order works. |
| `station[]` | the conveyor, **front first**: the first `n_pick` (5) can be picked, the rest up to `nb` (10) are preview. Sides and type only. |

Response:

| field | meaning |
|---|---|
| `status` | 0 OK, 1 PALLET_DONE (no box within reach has a legal place), 2 BAD_REQUEST |
| `message` | the reason, or warnings |
| `station_index` | which `station[]` box to pick |
| `placement` | centre x, y and bottom z (cm), sides as it will stand |
| `rotated` | true: the box is turned 90°, its length now lies along y |
| `pick_probability`, `place_probability` | the policy's confidence |
| `n_candidates` | how many legal places it chose from |

Feed `placement` straight back into `placed[]` on the next request.

### How cm become cells

Same as training: each box gets `box_pad_cm` (4 cm) on its length and width,
then every side is divided by `cell_cm` (4 cm) and rounded (`nearest`). A
50 x 30 x 28 box is 13 x 8 x 7 cells. A placed box's position is snapped
to the nearest cell. A reported `z_cm` more than `z_tolerance_cm` away from
where the box would actually rest gives a warning, and the rest height is used.

## Watching it live (the game's "Online game" tab)

After every request the node writes the step to `server.snapshot_file`
(`/tmp/pallet_rl_server/last_step.json` by default; `null` turns it off): the
pallet as the server rebuilt it, the conveyor, the picker's probability for
each box in reach, every legal candidate point with the packer's probability,
and what it chose, plus the points the rules turned down for the chosen box
(drawn red, each with the rule that did it). The file is replaced atomically.

`server.enable_extra_3d_viw: true` also brings every box within reach to the
front and scores it as if it were picked (its candidates, the packer's
probabilities, the points turned down), so the tab can show any box in reach,
not only the one picked. It slows each answer by about one more packer pass
per box in reach (~40 ms with the default rules on a desktop CPU).

The training repo's game shows it in its **Online game** tab, in one big 3D
view that follows each new step, with a slider back through the earlier ones
and the UR20 placing the chosen box:

```bash
cd ~/Projects/pallet_rl
python3 -m ar2l.viz.game --port 8096          # --online_file <path> if not the default
# open http://127.0.0.1:8096/#onl
```

## Settings

Everything is in [config/server.yaml](src/pallet_rl_server/config/server.yaml).
The node has one ROS parameter, `config`, which is the path to that file.

The rules must be the ones the model was trained with. At start-up the server
compares them with the training arguments stored in the checkpoint, and
`check_checkpoint` decides what a difference does: `warn` (default), `error`,
or `off`. To use another model, copy its `best.pt` into `models/`, point
`server.checkpoint` at it, and copy its rules from the run's `args.json`.

## Tests

```bash
.venv/bin/python -m pytest src/pallet_rl_server/test -q
```

`test_parity.py` plays real pallets with the original training env and
networks from the enclosing `pallet_rl` repo (or `PALLET_RL_REPO`). At every step it
rebuilds the same state from the boxes alone and checks that these match
exactly: the observations, both networks' logits, the pick, the placement,
and the cm answer. It covers six rule sets (column, mixed + soft mix, outline
step, touch + total support, cdrl without the arm). The test is skipped when
the repo is absent.

## Speed

On CPU, a request takes about 170 ms on average and 300 ms at p95. The
training env takes about 190 ms per step for the same decisions. The first
request after start-up is slower (about 0.9 s) while the arm-geometry cache
fills. That cache only remembers pure IK/capsule geometry per box pose, never
pallet state. Requests are served one at a time.
