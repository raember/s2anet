#!/bin/bash

DOCKER_CMD='nvidia-docker'
TAG='ghos/s2anet'
NAME='s2anet'
CODEBASE="$HOME/s2anet_uda/s2anet"
DATASET="$HOME/ds2_dense/"
NCPU=1
MEM=16
NGPU=1
