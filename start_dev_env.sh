#!/bin/bash

# Check if the Docker image exists
if [[ "$(docker images -q ml-inference:dev 2> /dev/null)" == "" ]]; then
    docker build -f docker/Dockerfile.dev -t ml-inference:dev .
fi

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    docker run \
        --name ml-processor \
        -it \
        --rm \
        --gpus all \
        -v $PWD:/ml-inference \
        ml-inference:dev
else
    docker run \
        --name ml-processor \
        -it \
        --rm \
        -v $PWD:/ml-inference \
        ml-inference:dev
fi