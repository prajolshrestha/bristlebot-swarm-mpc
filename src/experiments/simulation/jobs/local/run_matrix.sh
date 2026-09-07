#!/usr/bin/env bash
# Figs 6 and 7: deployed variant across four behaviors x three environments.
# Cluster equivalent: jobs/cluster/run_matrix.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
banner "matrix: Figs 6 and 7"

BEHAVIORS="${BEHAVIORS:-phototaxis phototaxis_orbital no_light phototaxis_orbital_contracting}"
cd "$CMP"
NB=$(echo $BEHAVIORS | wc -w); NN=$(echo $NS | wc -w)
echo "    $(( NB * NN * 3 * SEEDS )) runs"

# Emit the grid, then run it in parallel. One line per run keeps the failure
# mode obvious: a missing CSV maps to exactly one line here.
for b in $BEHAVIORS; do for n in $NS; do for o in 0 2 12; do
  for s in $(seq 0 $((SEEDS - 1))); do echo "$b $n $o $s"; done
done; done; done | xargs -P "$JOBS" -L1 bash -c '
  python matrix_wrapper.py --variant '"$VARIANT"' --behavior "$0" --n "$1" \
      --obstacles "$2" --reactive on --seed "$3" --steps '"$STEPS"' \
      --checkpoint-every 500 --flush-every 200 2>&1 | grep -E "DONE|Error" ' || true

python matrix_figures.py --collisions --summary --variant "$VARIANT" 2>&1 | tail -3
cd "$BEH"
for o in 0 2 12; do python fig_coordination.py --obstacles "$o" 2>&1 | tail -1; done
