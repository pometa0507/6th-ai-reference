FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basics
RUN apt-get update -y \
    && apt-get install build-essential \
    && apt-get install -y apt-utils git curl ca-certificates bzip2 tree htop wget \
    && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++ python3.8 python3.8-dev python3.8-distutils \
    && apt-get install -y libgl1-mesa-dev \
    && apt-get install -y cmake

# Clean apt cache
RUN apt-get autoremove -y &&\
  apt-get clean &&\
  rm -rf /usr/local/src/*

# Install python
RUN ln -sv /usr/bin/python3.8 /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py && \
    pip install -U pip

WORKDIR /root

# Install Boost geometry
RUN wget https://jaist.dl.sourceforge.net/project/boost/boost/1.68.0/boost_1_68_0.tar.gz && \
    tar xzvf boost_1_68_0.tar.gz && \
    cp -r ./boost_1_68_0/boost /usr/include && \
    rm -rf ./boost_1_68_0 && \
    rm -rf ./boost_1_68_0.tar.gz 

# Install Requirments
RUN mkdir /work
COPY requirements.txt /work
WORKDIR /work
RUN pip install --no-cache-dir -r requirements.txt

# Install spconv v1.2.1
WORKDIR /root
RUN git clone --depth 1 --recursive https://github.com/traveller59/spconv.git -b v1.2.1 && \
    cd ./spconv && \
    SPCONV_FORCE_BUILD_CUDA=1 python setup.py develop 
ENV PYTHONPATH=/root/spconv:$PYTHONPATH

WORKDIR /work

# execute jupyternotebook as a default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]