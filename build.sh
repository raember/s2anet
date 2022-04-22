#!/bin/bash

docker build -t s2anet:latest -f docker/Dockerfile .
docker build -t s2anetservice:latest -f docker/service/Dockerfile .
