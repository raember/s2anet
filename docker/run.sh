#!/bin/bash
# ./s2anet/docker/run.sh <docker-cmd> <tag> <name>

CODEBASE="$HOME/s2anet/s2anet"
DATASET="$HOME/ds2_dense"

echo "${1:-docker}" run -ditv "$CODEBASE":/s2anet -v "$DATASET":/s2anet/data --name "${3:-s2anet}" "${2:-s2anet}"
read -p "[ENTER] to run" _
"${1:-docker}" run -ditv "$CODEBASE":/s2anet -v "$DATASET":/s2anet/data --name "${3:-s2anet}" "${2:-s2anet}"
