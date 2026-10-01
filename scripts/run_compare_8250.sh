#!/bin/bash
# Big-grid half of the 8250 -> 10000 comparison.  Waits for the small-grid
# run (w_128_h_2_l_2_no_fine_tune) to finish so the two never share the GPU,
# then resumes w_128_h_2_l_2 on 120x100x160 at 1 cm with no box scale.
cd "$(dirname "$0")/.."
while pgrep -f "ar2l.train --name w_128_h_2_l_2_no_fine_tune" > /dev/null; do
    sleep 60
done
python3 -m ar2l.train --name w_128_h_2_l_2 --resume --iters 10000 \
    --width 128 --heads 2 --layers 2 \
    --pallet_cm 120 100 160 --cell_cm 1 --box_scale 1 --progress off \
    > results/logs/w_128_h_2_l_2_big.out 2>&1
