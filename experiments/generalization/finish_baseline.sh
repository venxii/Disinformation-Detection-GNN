#!/bin/zsh
# Step 1 completion: (a) rerun gat_full with the real seed (it was hard-wired to 7) on folds
# already written, then (b) finish missing folds. 3 jobs in parallel, 2 torch threads each.
cd "$(dirname $0)/../.."
R=artifacts/generalization/baseline; L=artifacts/generalization/logs
job() { # seed task
  local s=$1 t=$2
  if [[ $s != 7 ]]; then .venv/bin/python -m experiments.run_verifiers --task $t --seed $s --output $R/seed$s --models gat_full --merge --threads 2 >> $L/gatfix_${t}_seed$s.log 2>&1; fi
  .venv/bin/python -m experiments.run_verifiers --task $t --seed $s --output $R/seed$s --skip-done --threads 2 >> $L/baseline_${t}_seed$s.log 2>&1
}
for s in 7 13 21; do job $s binary & done; wait
for s in 13 21 42; do job $s 4class & done; wait
job 99 4class & job 42 binary & job 99 binary & wait
echo ALL DONE
