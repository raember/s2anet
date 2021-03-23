#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

ID="$1"
FP='deepscores_train.json'
echo "Fetching annotation $ID from train dataset"
if ! json="$(jq -ec ".annotations.\"$ID\"" "$FP")"; then
    echo "Fetching annotation $ID from test dataset"
    FP='deepscores_test.json'
    if ! json="$(jq -ec ".annotations.\"$ID\"" "$FP")"; then
        echo "Id $ID not found"
        exit 1
    fi
fi
python tools/deepscores_stats.py -g "$json"

echo "Use JS to retreive the new a_bbox coords:"
echo
echo -e "\033[1m[ggbApplet.getXcoord('A'), -ggbApplet.getYcoord('A'), ggbApplet.getXcoord('C'), -ggbApplet.getYcoord('C')].join(', ')\033[m"
echo
read -p "new a_bbox coords: " a_bbox

echo

echo "Use JS to retreive the new o_bbox coords:"
echo
echo -e "\033[1m[ggbApplet.getXcoord('E'), -ggbApplet.getYcoord('E'), ggbApplet.getXcoord('F'), -ggbApplet.getYcoord('F'), ggbApplet.getXcoord('G'), -ggbApplet.getYcoord('G'), ggbApplet.getXcoord('H'), -ggbApplet.getYcoord('H')].join(', ')\033[m"
echo
read -p "new o_bbox coords: " o_bbox

echo

echo "Saving a_bbox:"
echo "  [$a_bbox]"
echo "Saving o_bbox:"
echo "  [$o_bbox]"

read -p "Press [Enter] to continue"

echo "Saving a_bbox"
if ! jq -ec ".annotations.\"$ID\".a_bbox = [$a_bbox]" $FP > tmp.json; then
    echo "Failed saving new coords"
    exit 1
else
    mv tmp.json $FP
fi

echo "Saving o_bbox"
if ! jq -ec ".annotations.\"$ID\".o_bbox = [$o_bbox]" $FP > tmp.json; then
    echo "Failed saving new coords"
    exit 1
else
    mv tmp.json $FP
fi
echo "Done"
