#!/bin/zsh
# Step 5: RQ1–RQ3 (clean protocol) for one verifier configuration and seed.
#   [TASK=veracity] experiments/generalization/run_rq.sh <name> <verifier-root> <seed> [verifier-embedding-dir]
# Copies <verifier-root>/seed<S>/<task>/verifiers into artifacts/generalization/rq/<name>/seed<S>/ and reuses
# artifacts/lifecycle/stance_clean (per-fold stance models selected on validation; independent of the verifier).
set -e
cd "$(dirname $0)/../.."
name=$1 src=$2 seed=$3 vemb=${4:-} task=${TASK:-4class}
D=artifacts/generalization/rq/$name/seed$seed
mkdir -p $D/$task/verifiers
cp $src/seed$seed/$task/verifiers/*-threads.{npz,json} $src/seed$seed/$task/verifiers/*_dgtr.pt $D/$task/verifiers/
[[ -e $D/stance_clean ]] || ln -s ../../../../lifecycle/stance_clean $D/stance_clean
export OMP_NUM_THREADS=2
py=.venv/bin/python
$py -m experiments.run_commit_policies --task $task --root $D
$py -m experiments.run_lifecycle --task $task --root $D --clean
$py -m experiments.run_lifecycle --task $task --root $D --clean --summarize
$py -m experiments.run_injection --task $task --root $D --clean ${vemb:+--verifier-embedding-dir $vemb}
out=rq_results.json; [[ $task == 4class ]] || out=rq_results_$task.json
$py -m experiments.rq_analysis --root $D --tasks $task --output $out
echo "RQ DONE $name seed$seed $task"
