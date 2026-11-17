#!/bin/zsh
# Step 2: same LOEO runs as the baseline, on event-masked embeddings. 3 parallel jobs x 2 torch threads.
cd "$(dirname $0)/../.."
R=artifacts/generalization/masked; L=artifacts/generalization/logs
jobs_list=(); for t in 4class binary; do for s in 7 13 21 42 99; do jobs_list+="$t:$s"; done; done
run() { local t=${1%%:*} s=${1##*:}
  .venv/bin/python -m experiments.run_verifiers --task $t --seed $s --output $R/seed$s --embedding-dir data/nlp/embeddings/masked --skip-done --threads 2 >> $L/masked_${t}_seed$s.log 2>&1; }
i=0; for j in $jobs_list; do run $j & (( ++i % 3 == 0 )) && wait; done; wait
echo ALL DONE
