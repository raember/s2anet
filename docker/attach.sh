#!/bin/bash
# ./s2anet/docker/attach.sh [name]

. "$(dirname $0)/settings.sh"

cmd=("$DOCKER_CMD" exec -it "${1:-$NAME}" /bin/zsh)
echo "${cmd[@]}"
read -p "[ENTER] to run" _
"${cmd[@]}"
