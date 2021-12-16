#!/bin/bash

DOCKER_CMD='nvidia-docker'
TAG='urs/s2anet'
NAME='urs_s2anet'
CODEBASE="$HOME/rs/s2anet"
DATASET="$HOME/ds2_dense/"
NCPU=2
MEM=16
NGPU=1