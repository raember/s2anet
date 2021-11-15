# Required docker build context:
# .
# ├── mmcv/             # git clone https://github.com/tuggeluk/mmcv.git
# ├── obb_anns/         # git clone https://github.com/yvan674/obb_anns.git
# ├── s2anet/           # This repo
# └── authorized_keys   # Optional: ssh pub keys
# Build with:
# docker build -t <tag> -f s2anet/docker/Dockerfile .
# Run with:
# docker run -dp 2222:22 -v $HOME/s2anet/s2anet:/s2anet -v $HOME/ds2_dense:/s2anet/data --name <name> <tag>

ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y curl ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install OBB_Anns
# git clone https://github.com/yvan674/obb_anns.git
COPY /obb_anns /obb_anns
WORKDIR /obb_anns
RUN /opt/conda/bin/pip install -e .

# Install MMCV
# git clone https://github.com/tuggeluk/mmcv.git
COPY /mmcv /mmcv
WORKDIR /mmcv
ENV MMCV_WITH_OPS=1
RUN /opt/conda/bin/pip install -e .


# Install MMDetection deps
COPY /s2anet /s2anet_dummy
WORKDIR /s2anet_dummy
RUN /opt/conda/bin/conda install python=3.7
ENV FORCE_CUDA="1"
RUN /opt/conda/bin/conda install pytorch=1.3 torchvision cudatoolkit=${CUDA} -c pytorch
RUN /opt/conda/bin/pip install -r requirements.txt
RUN /opt/conda/bin/python setup.py develop
WORKDIR /s2anet_dummy/DOTA_devkit/polyiou
RUN swig -c++ -python csrc/polyiou.i
RUN /opt/conda/bin/python setup.py build_ext --inplace

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
