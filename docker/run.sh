#!/bin/bash
# ./s2anet/docker/run.sh [name [tag]]

. "$(dirname $0)/settings.sh"

CODEBASE="$HOME/s2anet/s2anet"
DATASET="$HOME/ds2_dense/"

echo "$DOCKER_CMD" run -ditv "$CODEBASE":/s2anet -v "$DATASET":/s2anet/data --name "${1:-$NAME}" "${2:-$TAG}"
read -p "[ENTER] to run" _
"$DOCKER_CMD" run -ditv "$CODEBASE":/s2anet -v "$DATASET":/s2anet/data --name "${1:-$NAME}" "${2:-$TAG}"
