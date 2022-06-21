#!/usr/bin/env bash

set -ex

IMAGE_TAG='ycho-isaac-gym'

# Figure out repository root.
SCRIPT_DIR="$( cd "$( dirname $(realpath "${BASH_SOURCE[0]}") )" && pwd )"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# Launch docker with the following configuration:
# * Display/Gui connected
# * Network enabled (passthrough to host)
# * Privileged
# * GPU devices visible
# * Current working git repository mounted at ${HOME}
# FIXME(ycho): hardcoded igibson package directory; is there a more robust workaround for this?
docker run -it --rm \
    --env DISPLAY \
    --env QT_X11_NO_MITSHM=1 \
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix \
    --volume="${HOME}/.Xauthority:/home/user/.Xauthority:rw" \
    --device=/dev/dri:/dev/dri \
    --mount type=bind,source=${REPO_ROOT},target="/home/user/$(basename ${REPO_ROOT})" \
    --mount type=bind,source=/home/jamie/isaacgym,target="/home/user/isaacgym" \
    --network host \
    --privileged \
    --gpus all \
    "$@" \
    "${IMAGE_TAG}"
