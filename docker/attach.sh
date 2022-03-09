#!/bin/bash
# ./s2anet/docker/attach.sh [name]

. "$(dirname $0)/settings.sh"

"$DOCKER_CMD" exec -it "${1:-$NAME}" /bin/zsh
