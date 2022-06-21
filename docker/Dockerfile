# syntax=docker/dockerfile:1.3
# TODO(ycho): Consider swapping base image depending on availability of graphics device.
ARG BASE_IMAGE=nvidia/cuda:11.0.3-devel-ubuntu20.04

FROM ${BASE_IMAGE} as base

# NOTE(ycho): Avoid questions during build process
ENV DEBIAN_FRONTEND=noninteractive
# NOTE(ycho): Avoid images getting bigger than necessary
ENV PIP_NO_CACHE_DIR=1

# NOTE(ycho): Re-declare `USE_CUDA` to bring within build scope.
ARG USE_CUDA=on
ARG USERNAME=user
ARG UID=1000
ARG GID=1000

# NOTE(ycho): Disable questions during build process
ENV DEBIAN_FRONTEND=noninteractive

# Install Core / networking packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    gnupg2 \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists*

# Install Packages for Building/Running iGibson.
# NOTE(ycho): `python3-dev` required for pybullet dependency.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    g++ \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists*

# Install packages for building `OpenSearch`.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libeigen3-dev \
    libopencv-dev \
    libfmt-dev && \
    rm -rf /var/lib/apt/lists*

# NOTE(ycho): libsm6, libice6, libnvidia-gl-460 required for EGL rendering
# FIXME(ycho): is it okay to hardcode `libnvidia-gl-460`?
# NOTE(ycho): libglib2.0-0 required for OpenCV; move it up from current pos.
# RUN sudo apt-get update && \
#     sudo apt-get install -y --no-install-recommends \
#     libsm6 \
#     libice6 \
#     libnvidia-gl-460 \
#     libglib2.0-0 && \
#     sudo rm -rf /var/lib/apt/lists*

# Setup GUI access enabled user.
# FIXME(ycho): Perhaps unnecessary since we need to run with
# --privileged option anyways (for network access).
RUN echo "useradd -m -s /bin/bash ${USERNAME}"
RUN [ $(getent group ${GID}) ] || groupadd -g ${GID} ${USERNAME} && \
    useradd -m -s /bin/bash ${USERNAME} -u ${UID} -g ${GID} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/${USERNAME} && \
    chmod 440 /etc/sudoers.d/${USERNAME} && \
    usermod -u ${UID} ${USERNAME} && \
    usermod -g ${GID} ${USERNAME}

# Add to `video` group (possibly needed to access /dev/dri)
RUN usermod -aG video ${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Add ~/.local/bin to path.
ENV HOME="/home/${USERNAME}"
ENV PATH="${PATH}:${HOME}/.local/bin"

# Update python deps.
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade wheel setuptools Cython

# Build iGibson.
# FIXME(ycho): `pyproject.toml` collision with editable installs.
# Deal with this at some point.
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxext-dev \
    && sudo rm -rf /var/lib/apt/lists*

# Install additional python packages for building `opensearch`.
RUN python3 -m pip install --upgrade \
    cmake \
    cmake-build-extension \
    pybind11 \
    shapely \
    chama \
    pandas \
    scikit-image \
    triangle

RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    libeigen3-dev \
    python3-tk \
    && sudo rm -rf /var/lib/apt/lists*

RUN python3 -m pip install --upgrade \
    line_profiler \
    cachetools

# NOTE(ycho): Fork of SimpleParsing with frozenset support.
RUN python3 -m pip install --upgrade \
    git+https://github.com/yycho0108/SimpleParsing.git@serialize-frozenset

RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    gdb \
    && sudo rm -rf /var/lib/apt/lists*

# OMPL dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    libboost-filesystem-dev \
    libboost-python-dev \
    libboost-numpy-dev \
    libboost-serialization-dev \
    libboost-program-options-dev \
    && sudo rm -rf /var/lib/apt/lists*

# Install OMPL.
WORKDIR /home/${USERNAME}
RUN git clone https://github.com/ompl/ompl.git && \
    cd ompl && \
    mkdir -p build/Release && \
    cd build/Release && \
    cmake ../../ -DPYTHON_EXEC=/usr/bin/python3 && \
    make -j4 && \
    sudo make install

# Install GNU Parallel.
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    parallel \
    && sudo rm -rf /var/lib/apt/lists*

# Install libyaml-cpp-dev.
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    libyaml-cpp-dev \
    && sudo rm -rf /var/lib/apt/lists*

RUN echo 'export PATH="$HOME/.local/bin:${PATH}"' >> "${HOME}/.bashrc"
# export `CUDA_HOME` env variable to .bashrc.
RUN if [ "${USE_CUDA}" = "on" ]; then \
    printf 'export CUDA_HOME=/usr/local/cuda\nexport LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"\nexport PATH="${PATH}:${CUDA_HOME}/bin"' >> "${HOME}/.bashrc"; \
    fi
RUN python3 -m pip install \
    torch==1.10.1+cu111 \
    torchvision==0.11.2+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html
CMD ["/bin/bash"]
