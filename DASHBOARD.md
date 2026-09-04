# Reading the dashboard

`python -m src.viz.dashboard --port 8080` → http://127.0.0.1:8080

Numbers in the "now" column are from the live CUT-2 run at 5.76M steps, for reference.

---

## Top strip

| | Means | Good |
|---|---|---|
| **Progress bar** | steps done / target, and time left from the *recent* rate (not the lifetime average) | — |
| **Pipeline queue** | every scripted stage; blue = running, green = done, grey = queued | — |

---

## Tier 1 — is it actually working?

These two are the result. Everything else explains them.

### 1. Space utilisation `space_util`
Fraction of the bin volume filled when the episode ends. **This is the paper's headline metric.**

- **Want:** rising, then flattening. Paper's CUT-2 target is **66.9%**.
- **Now:** 36.6% → **52.4%**
- **Bad:** flat from early on (not learning), or a sudden drop (policy collapse — check KL).
- A slow-but-still-rising curve is fine. Stop when it's been flat for several million steps.

### 2. Items packed / bin `n_items`
How many boxes fit before the episode ended.

- **Want:** rising with utilisation. Paper: **17.5** items.
- **Now:** 9.6 → **13.6**
- Tracks utilisation closely. If utilisation rises while item count *falls*, the agent is
  favouring a few big boxes — not wrong, but worth noticing.

---

## Tier 2 — is the constrained-DRL scheme working?

This is the paper's actual contribution. If these are bad, Tier 1 can't get good.

### 3. Invalid-action rate `invalid_rate`, `mask_fpr` *(log scale)*
- `invalid_rate` — how often the agent picked a loading position that is physically illegal.
- `mask_fpr` — how often the mask predictor labels an illegal position as legal (false-feasible).

- **Want:** both falling toward **< 1%**. The paper claims **99.5% legit** placements, i.e. ~0.5%.
- **Now:** 5.9% → **2.1%** invalid, 3.1% → **1.5%** false-feasible.
- `mask_fpr` should sit *below* `invalid_rate`, and it does. If `invalid_rate` stays high while
  `mask_fpr` is near zero, the predictor is fine but the projection isn't being applied.
- **Bad:** plateau above ~5%. That's what a too-short run looks like, and it silently caps
  utilisation because episodes keep dying early.

### 4. Mask predictor `mask_acc`, `mask_rec`
Accuracy and recall of the predicted feasibility mask vs ground truth.

- **Want:** accuracy **> 99%**, recall close behind.
- **Now:** 97.6% → **99.1%** accuracy, 94.2% → **97.6%** recall.
- **Recall matters separately:** low recall means the predictor is calling *legal* positions
  illegal, so the agent never explores them — it caps utilisation without ever showing up as an
  invalid action. Watch the gap between the two lines.

### 5. Episode end reason `frac_seq_end`, `frac_no_feasible`, `frac_invalid_end`
Why episodes stopped.

- `frac_invalid_end` — agent chose an illegal spot. **This is the failure you want to eliminate.**
- `frac_no_feasible` — genuinely no legal spot left. **This is the real problem being solved.**
- `frac_seq_end` — packed the entire sequence. Expected to stay ~0 (the paper packs 17.5 of ~26).

- **Want:** the two swap over — invalid-end falls, no-feasible rises.
- **Now:** invalid-end 60.8% → **29.4%**, no-feasible 39.2% → **70.7%**. This crossover is the
  clearest single sign the constraint machinery is doing its job: the agent has stopped making
  illegal moves and is now just running out of room, which is the actual packing difficulty.

### 6. Mask loss (MSE) `mask`
Supervised loss of the mask predictor.

- **Want:** drops fast, then flat and small.
- **Now:** 0.019 → **0.0074**. Healthy.
- **Bad:** rising — the state distribution has shifted somewhere the predictor never trained.

### 7. E_inf `einf` *(log scale)*
Probability mass the policy still puts on infeasible positions (paper Eq. 2 penalty term).

- **Want:** decaying toward 0.
- **Now:** 0.061 → **0.023**
- This is the "soft" version of the invalid rate — it moves before the invalid rate does, so
  it's the earlier warning signal of the two.

### 8. Feasibility entropy `entropy`
Exploration, measured **only over legal actions** (the paper's FE term).

- **Want:** high early, decaying smoothly, settling **well above 0**.
- **Now:** 0.347 → **0.289**
- **Bad — collapse to ~0:** the policy became deterministic too early and stopped exploring;
  utilisation will flatten prematurely. Raise `w_entropy` (ψ, default 0.01).
- **Bad — stays high and flat:** it's not committing to anything. Lower ψ.

---

## Tier 3 — PPO health (nothing to do with packing)

These say whether the *optimiser* is stable. Read them only when Tier 1/2 misbehave.

### 9. Actor loss `actor`
**Do not read this as progress.** It's a clipped surrogate objective, not an error. It hovers
near zero and is often negative (**now: −0.008**). Only its *variance* is informative — wild
spikes mean unstable updates.

### 10. Critic loss `critic`
Value-function regression error.

- **Rising early is normal and expected** — as the policy improves, returns grow, so the target
  the critic is chasing gets bigger. Yours rose to ~0.46 then settled to **0.295**.
- **Want:** rise, peak, then decline or plateau.
- **Bad:** unbounded growth — the critic has lost the plot; lower the learning rate.

### 11. Approx KL / clip fraction `approx_kl`, `clipfrac`
How far each update moves the policy.

- **Want:** `approx_kl` roughly **0.01–0.05**, `clipfrac` around **0.1–0.2**, both stable.
- **Now:** KL **0.065**, clipfrac **0.118**. KL is a little on the high side but trending down
  (was 0.136) — fine here because the linear LR decay keeps shrinking it. If it climbed past
  ~0.1 *and* utilisation stalled, I'd cut the learning rate.
- **Bad:** KL spiking → the policy is jumping too far and can collapse; clipfrac near 0 → the
  updates are so tiny nothing is being learned.

### 12. Throughput `fps`
Environment steps per second, lifetime average.

- **Want:** flat. **Now: ~4,220** (the paper quotes 2,000).
- A gradual decline usually means something else started competing for the GPU.

---

## Quick triage

| Symptom | Most likely cause |
|---|---|
| Utilisation flat, invalid rate > 5% | Mask predictor undertrained — just needs more steps |
| Utilisation flat, mask acc > 99%, entropy ≈ 0 | Premature convergence — raise ψ (`w_entropy`) |
| Utilisation flat, mask recall low | Predictor is over-rejecting legal spots |
| Utilisation drops suddenly | Policy collapse — check `approx_kl`, lower `lr` |
| `frac_invalid_end` stuck high | Projection not effective — check MC is on (`--no-mc` off) |
| Critic loss growing without bound | Lower `lr` |
