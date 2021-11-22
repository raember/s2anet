# Required docker build context:
# .
# ├── mmcv/             # git clone https://github.com/tuggeluk/mmcv.git
# ├── obb_anns/         # git clone https://github.com/yvan674/obb_anns.git
# ├── s2anet/           # This repo https://github.com/raember/s2anet.git
# └── authorized_keys   # Optional: ssh pub keys
# Build with:
# docker build -t <tag> -f s2anet/docker/Dockerfile .
# Run with:
# docker run -dp 2222:22 -v $HOME/s2anet/s2anet:/s2anet -v $HOME/ds2_dense:/s2anet/data --name <name> <tag>

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

# Install OBB_Anns
FROM build-pytorch AS build-obb_anns
WORKDIR /
COPY /obb_anns /obb_anns
#RUN git clone https://github.com/yvan674/obb_anns.git
RUN pip install -e /obb_anns

# Install MMDetection deps
FROM build-obb_anns AS build-mmdet
WORKDIR /
COPY /s2anet /s2anet_dummy
#RUN git clone https://github.com/raember/s2anet.git s2anet_dummy
RUN conda install python=3.8
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
WORKDIR /s2anet
# /s2anet/data -> DS2 dataset
WORKDIR /s2anet/data

# Setup ssh port forwarding
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir -p /root/.ssh
ADD authorized_keys /root/.ssh/authorized_keys
RUN chown root:root /root/.ssh/authorized_keys
RUN mkdir /var/run/sshd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
