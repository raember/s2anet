# Required docker build context:
# .
# ├── mmcv/             # git clone https://github.com/tuggeluk/mmcv.git
# ├── obb_anns/         # git clone https://github.com/yvan674/obb_anns.git
# └── s2anet/           # This repo https://github.com/raember/s2anet.git
# Build with:
# docker build -t <tag> -f s2anet/docker/Dockerfile .
# Run with:
# docker run -dv $HOME/s2anet/s2anet:/s2anet -v $HOME/ds2_dense:/s2anet/data --name <name> <tag>

ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

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
#RUN git clone https://github.com/yvan674/obb_anns
#COPY /obb_anns /obb_anns
RUN git clone https://github.com/yvan674/obb_anns.git
WORKDIR /obb_anns
RUN python3.8 setup.py develop
#RUN pip install -e .

# Install MMDetection deps
FROM build-obb_anns AS build-mmdet
WORKDIR /
#COPY /s2anet /s2anet_dummy
#RUN git clone https://github.com/raember/s2anet.git s2anet_dummy
RUN conda install cython -y && conda clean --all
RUN git clone https://github.com/raember/s2anet.git /s2anet_2
WORKDIR /s2anet_2
#RUN pip install --no-cache-dir -e .
RUN python setup.py develop

ENV FORCE_CUDA="1"
RUN conda install pytorch=${PYTORCH} torchvision cudatoolkit=${CUDA} -c pytorch
RUN pip install -r /s2anet_dummy/requirements.txt
RUN pip install wandb
# Install custom polyiou
FROM build-mmdet AS build-polyiou
WORKDIR /s2anet_dummy/DOTA_devkit/polyiou
RUN swig -c++ -python csrc/polyiou.i
RUN python setup.py build_ext --inplace

# Install MMCV
FROM build-polyiou AS build-mmcv
WORKDIR /
#COPY /mmcv /mmcv
RUN git clone https://github.com/tuggeluk/mmcv.git
ENV MMCV_WITH_OPS=1
RUN pip install -e /mmcv

FROM build-mmcv AS build-aftermath

# Setup s2anet for disk sharing
# /s2anet -> code base
WORKDIR /s2anet_2
# /s2anet/data -> DS2 dataset
WORKDIR /s2anet_2/data