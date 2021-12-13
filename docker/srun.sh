#!/bin/bash
# ./s2anet/docker/srun.sh [jobname]

. "$(dirname $0)/settings.sh"

echo srun --job-name="${1:-NAME}" --pty --ntasks=1 --cpus-per-task="$NCPU" --mem="${MEM}G" --gres="gpu:$NGPU" $SHELL
read -p "[ENTER] to run" _
srun --job-name="${1:-NAME}" --pty --ntasks=1 --cpus-per-task="$NCPU" --mem="${MEM}G" --gres="gpu:$NGPU" $SHELL
