#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/gridview/slurm/bin:$PATH"
if [[ $# -lt 3 || $# -gt 4 || "$1" != --check && "$1" != --submit ]]; then
    echo "Usage: bash submit.sh --check|--submit WORK PARTITION [ACCOUNT]" >&2
    exit 2
fi
mode=$1
work=$(cd "$2" && pwd)
partition=$3
account=${4:-}
py=${CS290_PYTHON:-"$PWD/.venv/bin/python"}
if [[ -e "$work/submission" ]]; then
    echo "Existing submission record at $work/submission. Inspect it before submitting again." >&2
    exit 2
fi
"$py" pilot.py preflight --work "$work"
mkdir -p "$work/logs"
args=(--job-name=cs290-rain --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G
      --time=06:00:00 --gres="gpu:${CS290_A40_TYPE:-NVIDIAA40}:1" --partition="$partition"
      --output="$work/logs/%j.log")
if [[ -n "$account" ]]; then
    args+=(--account="$account")
fi
sbatch --test-only "${args[@]}" "$PWD/job.sh" "$PWD" "$work"
if [[ "$mode" == --submit ]]; then
    mkdir "$work/submission"
    sbatch --parsable "${args[@]}" "$PWD/job.sh" "$PWD" "$work" | tee "$work/submission/job_id"
fi
