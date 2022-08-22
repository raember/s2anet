#!/bin/bash
# ./docker/build.sh [type]
# type = {"", "dev", "prod"}

. "$(dirname $0)/settings.sh"

df="$1"
if ! [[ "$df" == "" ]]; then
    df="-$df"
fi

dockerfile="docker/Dockerfile$df"
if ! [[ -f "$dockerfile" ]]; then
    echo "Docker file does not exist! Only allowed types: \"\", \"dev\", \"prod\""
    exit 1
fi

cmd=(docker build -f "$dockerfile" . -t "$TAG$df")

echo "${cmd[@]}"
read -p "[ENTER] to run" _
"${cmd[@]}"
