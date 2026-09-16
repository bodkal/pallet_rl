"""What the permutation-based attacker actually does.

AR2L Sec. 4.1 claims the attacker "appears to prefer smaller items when
constructing harder instances as the number of observable items increases".
This view measures that: for a trained attacker it plots the size distribution
of the items it promotes against the nominal distribution, how far down the
conveyor it reaches, and how much utilisation it destroys, per N_B.

    python3 -m ar2l.viz.attack --runs att_pct_nb5 att_pct_nb10 att_pct_nb20
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..env import BPPBatch
from ..evaluate import load_nets, metrics
from ..model import sample
from ..ppo import to_torch
from . import agents as A


@torch.no_grad()
def profile(pack, attacker, nb, seqs, device="cuda", stability="com"):
    """Per-step record of which conveyor slot and which item size was promoted."""
    env = BPPBatch(len(seqs), nb=nb, n_items=seqs.shape[1], stability=stability)
    env.reset(seqs)
    slots, chosen, steps = [], [], []
    t = 0
    while not env.done.all():
        win, wmask = env.window()
        alive = ~env.done
        o = to_torch(env.obs_cb(), device)
        idx, _, _ = sample(attacker.logits(o), greedy=True)
        idx = idx.cpu().numpy()
        slots.append(idx[alive])
        chosen.append(win[np.arange(len(seqs)), idx][alive])
        steps.append(np.full(int(alive.sum()), t))
        t += 1
        env.permute(idx)
        o = to_torch(env.obs(), device)
        a, _, _ = sample(pack.logits(o), greedy=True)
        env.step(a.cpu().numpy())
    return {"slot": np.concatenate(slots),
            "chosen": np.concatenate(chosen),
            "step": np.concatenate(steps),
            "util": env.utilization(), "items": env.n_packed.astype(float)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='runs holding an attacker, e.g. att_pct_nb10')
    ap.add_argument('--data', default='data/discrete_test.npy')
    ap.add_argument('--n_inst', type=int, default=512)
    ap.add_argument('--root', default='.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='results/attacker_behaviour.png')
    a = ap.parse_args(argv)

    seqs = np.load(os.path.join(a.root, a.data))[: a.n_inst]
    # an item the attacker passes over stays in the window and would be
    # counted again, so the baseline is the dataset itself, not what was seen
    base = float(seqs.reshape(-1, 3).prod(1).mean())
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    summary = {}
    for name in a.runs:
        info = A.run_info(name)
        nets = load_nets(info["ckpt"], a.device, ("pack", "attacker"))
        nb = info["nb"]
        pr = profile(nets["pack"], nets["attacker"], nb, seqs, a.device)
        vol = pr["chosen"].prod(1)
        lab = f"$N_B$={nb}"
        axes[0].hist(vol, bins=np.arange(0, 130, 6), density=True, histtype="step",
                     lw=1.6, label=lab)
        axes[1].hist(pr["slot"], bins=np.arange(-0.5, max(nb, 2) + 0.5),
                     density=True, histtype="step", lw=1.6, label=lab)
        summary[name] = {"nb": nb, "mean_promoted_volume": float(vol.mean()),
                         "mean_item_volume": base,
                         "mean_slot": float(pr["slot"].mean()),
                         **metrics(pr["util"], pr["items"])}
        axes[2].scatter([nb], [summary[name]["uti"]], s=44)
        axes[2].annotate(name, (nb, summary[name]["uti"]), fontsize=7,
                         textcoords="offset points", xytext=(4, 4))
        # the paper's claim: small items get promoted early, to starve the
        # policy of the supportive planes it wants at the bottom of the bin
        T = int(pr["step"].max()) + 1
        mean_t = np.array([vol[pr["step"] == i].mean() if (pr["step"] == i).any()
                           else np.nan for i in range(T)])
        axes[3].plot(np.arange(T), mean_t, lw=1.6, label=lab)
    axes[0].axvline(base, color="k", ls="--", lw=1, label="nominal mean")
    axes[0].set_xlabel("volume of the promoted item"); axes[0].set_ylabel("density")
    axes[0].set_title("what the attacker reaches for"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("conveyor slot promoted"); axes[1].set_title("how far it reaches")
    axes[1].legend(fontsize=7)
    axes[2].set_xlabel("$N_B$"); axes[2].set_ylabel("utilisation under attack (%)")
    axes[2].set_title("damage done"); axes[2].grid(alpha=.3)
    axes[3].axhline(base, color="k", ls="--", lw=1, label="nominal mean")
    axes[3].set_xlabel("step in the episode")
    axes[3].set_ylabel("volume of the promoted item")
    axes[3].set_title("small items first?"); axes[3].grid(alpha=.3)
    axes[3].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(a.root, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    json.dump(summary, open(out.replace('.png', '.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    print('->', out)


if __name__ == "__main__":
    main()
