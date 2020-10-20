FROM ubuntu:18.04

ENV PYTHON_VERSION 3.6.4
ENV TF_PACKAGE tensorflow
ENV TF_PACKAGE_VERSION 2.3.1

# ubuntu 基础软件包
RUN (apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        software-properties-common \
        curl \
        wget \
        git \
        pkg-config \
        zlib1g-dev \
        openssh-client \
        g++ unzip zip)

# 时区设置为上海
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y tzdata
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

# https://github.com/pypa/pip/issues/4924
RUN mv /usr/bin/lsb_release /usr/bin/lsb_release.bak


# 安装 Python
WORKDIR /tmp/
RUN (wget -P /tmp https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz)
RUN tar -zxvf Python-$PYTHON_VERSION.tgz
WORKDIR /tmp/Python-$PYTHON_VERSION
RUN apt-get update -y && apt-get upgrade -y && apt-get dist-upgrade -y

# python编译所需要的依赖
RUN apt-get install -y --no-install-recommends libbz2-dev libncurses5-dev libgdbm-dev libgdbm-compat-dev liblzma-dev libsqlite3-dev libssl-dev openssl tk-dev uuid-dev libreadline-dev
RUN apt-get install -y --no-install-recommends libffi-dev
RUN ./configure --prefix=/usr/local/python3 && \
        make && \
        make install

# 配置python3 python pip pip3
RUN update-alternatives --install /usr/bin/python python /usr/local/python3/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/python3/bin/pip3 1
RUN update-alternatives --config python
RUN update-alternatives --config pip
RUN pip install --upgrade pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/python3/bin/python3 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/local/python3/bin/pip3 1
RUN update-alternatives --config python3
RUN update-alternatives --config pip3


# 常用py packages
RUN (pip --no-cache-dir install \
        setuptools \
        wheel \
        cython \
        pytest \
        pytest-cov \
        prometheus_client \
        coverage \
        h5py \
        pyzmq \
        scipy \
        pandas \
        sklearn \
        matplotlib \
        Pillow \
        glog \
        boto3 \
        hanziconv \
        PyYAML \
        lxml \
        mysql-connector \
        uuid \
        uwsgi \
        bs4 \
        pybind11 \
        redis \
        pyzmq \
	    google==2.0.3\
        minio \
        tqdm \
        rarfile)

# 机器学习常用包
RUN pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}
RUN (pip --no-cache-dir install \
    tensorboard==2.3.0\
    tensorflow-serving-api)

# jupyter
COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc
RUN pip install --no-cache-dir jupyter matplotlib
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
RUN pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0
RUN jupyter serverextension enable --py jupyter_http_over_ws


# maddpg 依赖环境
# install multiagent-particle-envs
RUN mkdir /root/github
ENV CODE_DIR /root/github
WORKDIR  $CODE_DIR
RUN cd $CODE_DIR
RUN rm -rf multiagent-particle-envs
RUN git clone https://github.com/iminders/multiagent-particle-envs.git
# Clean up pycache and pyc files
RUN cd $CODE_DIR/multiagent-particle-envs && rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .


# Install Open MPI
# download realese version from official website as openmpi github master is not always stable
ARG OPENMPI_VERSION=openmpi-4.0.0
ARG OPENMPI_DOWNLOAD_URL=https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget ${OPENMPI_DOWNLOAD_URL} && \
    tar zxf ${OPENMPI_VERSION}.tar.gz && \
    cd ${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

RUN pip install --no-cache-dir mpi4py

# clean
RUN rm -rf /tmp/* && \
rm -rf /var/lib/apt/lists/* && \
rm -rf /root/.cache/pip

WORKDIR $CODE_DIR

CMD ["bin/bash"]
