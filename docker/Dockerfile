ARG PYTORCH="1.3.0"
ARG CUDA="10.0"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y curl ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
#RUN conda clean --all
#RUN conda install cython

## Install OBB_Anns
#WORKDIR /
#RUN git clone https://github.com/yvan674/obb_anns.git
#WORKDIR /obb_anns
##RUN python setup.py develop
#RUN pip install -e .

# Install MMCV
WORKDIR /
RUN curl "https://codeload.github.com/open-mmlab/mmcv/tar.gz/v0.2.14" | tar -xz
WORKDIR /mmcv-0.2.14
ENV MMCV_WITH_OPS=1
RUN /opt/conda/bin/pip install -e .


# Install MMDetection deps
WORKDIR /
RUN git clone https://github.com/csuhan/s2anet.git dummdet
WORKDIR /dummdet
RUN /opt/conda/bin/conda install python=3.7
ENV FORCE_CUDA="1"
RUN /opt/conda/bin/conda install pytorch=1.3 torchvision cudatoolkit=${CUDA} -c pytorch
RUN /opt/conda/bin/pip install -r /dummdet/requirements.txt
RUN /opt/conda/bin/python setup.py develop
WORKDIR /dummdet/DOTA_devkit/polyiou
RUN swig -c++ -python csrc/polyiou.i
RUN /opt/conda/bin/python setup.py build_ext --inplace
# Install s2anet
WORKDIR /s2anet
#RUN conda create -n s2anet python=3.7 -y


# Make mountpoint for disk sharing
# /mmdetection -> code base
# /workspace -> DS2 dataset
#WORKDIR /mmdetection/data
#RUN ln -s /workspace ds2_dense

# Setup ssh port forwarding
RUN apt-get update && apt-get install -y openssh-server
ADD .ssh.tar.gz /tmp/
RUN mkdir -p /root/.ssh \
        && mv /tmp/.ssh/authorized_keys /root/.ssh/authorized_keys \
        && rm -rf /tmp/.ssh .ssh.tar.gz \
        && chown root:root /root/.ssh/authorized_keys
RUN mkdir /var/run/sshd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

# cd /mmdetection
# /opt/conda/bin/pip install -e .
# /opt/conda/bin/pip install future tensorboard
