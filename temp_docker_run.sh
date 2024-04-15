#!/bin/bash

# Check if the Docker image exists
# if [[ "$(docker images -q bay-ml-processing 2> /dev/null)" == "" ]]; then
docker build -f dockerfiles/Dockerfile.infer -t ml-infer .
# fi

docker run \
    --name ml-infer-container \
    -it \
    --rm \
    --gpus all \
    ml-infer:latest