# Online 3D-BPP with Buffer (arXiv:2208.07123) - reproduction

Average packed items / space utilisation on the 100 held-out test sequences of each dataset, 10x10x10 bin.

### Comparable to the paper - no knowledge of the upcoming sequence

| Dataset | b | k | Heuristics [8] | Model-free [2] | Paper (Ours) | **This run, policy only** | % of paper |
|---|---|---|---|---|---|---|---|
| CUT-1 | 1 | 0 | 15.2 / 59.8% | 19.1 / 73.4% | 21.3 / 83.4% | **75.1%** | 90% |
| CUT-2 | 1 | 0 | 17.3 / 61.2% | 17.5 / 66.9% | 18.0 / 69.9% | **67.9%** | 97% |
| RS | 1 | 0 | 13.8 / 54.3% | 12.2 / 50.5% | 13.1 / 53.1% | **52.2%** | 98% |
| CUT-2 | 2 | 0 | - | - | 20.2 / 71.5% | **67.9%** | 95% |
| CUT-2 | 3 | 0 | - | - | 21.8 / 77.1% | **69.6%** | 90% |
| CUT-2 | 5 | 0 | - | - | - | **68.0%** | - |
| CUT-2 | 7 | 0 | - | - | - | **70.1%** | - |

### With search over the known sequence - an oracle, not comparable

| Dataset | MCTS, real stream | MCTS, shuffled stream | MCTS, independent draw | policy only | what search is worth without the oracle |
|---|---|---|---|---|---|
| CUT-1 | **89.8%** | 77.5% | 71.5% | 75.1% | **-3.6** |
| CUT-2 | **80.6%** | 70.4% | 67.7% | 67.9% | **-0.2** |
| RS | **63.2%** | 50.8% | 51.0% | 52.2% | **-1.2** |
| CUT-2 | **80.6%** | 70.4% | 67.7% | 67.9% | **-0.2** |
| CUT-2 | **80.6%** | 70.4% | 67.7% | 67.9% | **-0.2** |
| CUT-2 | **80.6%** | 70.4% | 67.7% | 67.9% | **-0.2** |
| CUT-2 | **80.6%** | 70.4% | 67.7% | 67.9% | **-0.2** |
