FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.20
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
         libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=3.5 numpy pyyaml scipy ipython mkl && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update
RUN apt-get install -y --no-install-recommends mercurial
RUN pip install cython
RUN BACKEND=cuda pip install git+https://github.com/clab/dynet#egg=dynet
RUN apt-get -y install vim
RUN pip install jupyter
#RUN pip install --no-cache-dir http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl

# Required for viewing logged events
# RUN pip install --no-cache-dir --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp35-cp35m-linux_x86_64.whl


WORKDIR /workspace
RUN chmod -R a+w /workspace

