# Required docker for build context:
# .
# ├── mmcv/             # git clone https://github.com/tuggeluk/mmcv.git
# ├── obb_anns/         # git clone https://github.com/raember/obb_anns.git
# └── s2anet/           # This repo https://github.com/raember/s2anet.git
# Build with:
# docker build -t <tag> -f s2anet/docker/Dockerfile .
# Run with:
# docker run -dv $HOME/s2anet/s2anet:/s2anet -v $HOME/ds2_dense:/s2anet/data --name <name> <tag>

ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel AS build-pytorch

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y curl ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN conda install python=3.8

# Install OBB_Anns
FROM build-pytorch AS build-obb_anns
WORKDIR /
COPY /obb_anns /obb_anns
RUN git clone https://github.com/raember/obb_anns.git
WORKDIR /obb_anns
RUN python setup.py develop

# Install MMDetection/S2ANet deps
FROM build-obb_anns AS build-mmdet
RUN conda install cython -y && conda clean --all
COPY /s2anet /s2anet_dummy
RUN git clone https://github.com/raember/s2anet.git s2anet_dummy
WORKDIR /s2anet_dummy
ENV FORCE_CUDA="1"
RUN conda install pytorch=${PYTORCH} torchvision cudatoolkit=${CUDA} -c pytorch
RUN pip install -r /s2anet_dummy/requirements.txt
RUN pip install wandb

# Install custom polyiou
FROM build-mmdet AS build-polyiou
WORKDIR /s2anet_dummy/DOTA_devkit/polyiou
RUN swig -c++ -python csrc/polyiou.i
RUN python setup.py build_ext --inplace

# Install S2ANet
FROM build-polyiou AS build-s2anet
RUN pip install --no-cache-dir -e /s2anet_dummy

# Install MMCV
FROM build-s2anet AS build-mmcv
WORKDIR /
#COPY /mmcv /mmcv
RUN git clone https://github.com/tuggeluk/mmcv.git
# Remove old mmcv
RUN pip uninstall mmcv -y
ENV MMCV_WITH_OPS=1
RUN pip install -e /mmcv

# Setup s2anet for disk sharing
FROM build-mmcv AS build-aftermath
# /s2anet/data -> DS2 dataset
WORKDIR /s2anet/data
# /s2anet -> code base
WORKDIR /s2anet
# Use this as the landing dir when attaching a shell to the running container

# Setup miscellaneous things for usability
FROM build-aftermath AS build-custom
RUN apt update
RUN apt install tmux vim zsh -y
RUN echo | RUNZSH=no sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN sed 's/robbyrussell/crcandy/g' -i ~/.zshrc
# Don't forget to
# python -m wandb login <token>