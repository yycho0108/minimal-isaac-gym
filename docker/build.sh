#!/usr/bin/env bash

set -ex

IMAGE_TAG='ycho-isaac-gym'

# NOTE(ycho): Set context directory relative to this file.
CONTEXT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"

# Check for existence of Nvidia GPU.
# FIXME(ycho): NOT the most rigorous check in the world.
USE_CUDA="off"
if [[ $(lshw -C display 2>/dev/null | grep vendor | awk '{print tolower($0)}') =~ nvidia ]]; then
    USE_CUDA="on"
fi

## Build docker image.
DOCKER_BUILDKIT=1 docker build --progress=plain \
    --build-arg USE_CUDA="${USE_CUDA}" \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    -t "${IMAGE_TAG}" -f ${CONTEXT_DIR}/Dockerfile ${CONTEXT_DIR}
