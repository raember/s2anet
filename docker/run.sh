#!/bin/bash
# ./s2anet/docker/run.sh [name [tag]]

. "$(dirname $0)/settings.sh"

echo "$DOCKER_CMD" run --shm-size=16g -ditv "$CODEBASE":/s2anet -v "$DATASET":/s2anet/data --name "${1:-$NAME}" "${2:-$TAG}"
read -p "[ENTER] to run" _
"$DOCKER_CMD" run --shm-size=16g -ditv "$CODEBASE":/s2anet -v "$DATASET":/s2anet/data -p 8800:22 --name "${1:-$NAME}" "${2:-$TAG}"
