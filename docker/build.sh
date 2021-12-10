#!/bin/bash
# ./s2anet/docker/build.sh [tag]

. "$(dirname $0)/settings.sh"

DOCKER_CONTEXT="$(pwd)"
echo "Docker build context:"
echo "$DOCKER_CONTEXT"
echo " ."
for f in $DOCKER_CONTEXT/.*; do
    if [[ $(basename $f) == "." ]] || [[ $(basename $f) == ".." ]]; then
        continue
    fi
    echo " └── $(basename $f)"
done
seems_fine=
for f in $DOCKER_CONTEXT/*; do
    echo " └── $(basename $f)"
    if [[ $(basename $f) == "s2anet" ]]; then
        seems_fine=1
    fi
done

if [[ -z $seems_fine ]]; then
    echo "The docker context does not seem fine. Make sure to be one level above the s2anet root directory"
    exit 1
fi

echo "$DOCKER_CMD" build -t "${1:-$TAG}" -f s2anet/docker/Dockerfile .
read -p "[ENTER] to run" _
"$DOCKER_CMD" build -t "${1:-$TAG}" -f s2anet/docker/Dockerfile .
