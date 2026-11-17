#!/bin/zsh
# 5-seed veracity baseline (same pipeline as 4class/binary), then the clean RQ pipeline per seed.
cd "$(dirname $0)/../.."
R=artifacts/generalization/baseline; L=artifacts/generalization/logs
v() { .venv/bin/python -m experiments.run_verifiers --task veracity --seed $1 --output $R/seed$1 --skip-done --threads 2 >> $L/baseline_veracity_seed$1.log 2>&1; }
v 7 & v 13 & v 21 & wait; v 42 & v 99 & wait
rq() { TASK=veracity ./experiments/generalization/run_rq.sh baseline $R $1 > $L/rq_baseline_veracity_seed$1.log 2>&1; }
rq 7 & rq 13 & rq 21 & wait; rq 42 & rq 99 & wait
echo ALL DONE
