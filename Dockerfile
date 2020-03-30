# https://github.com/iminders/mpi/blob/master/Dockerfile
from registry.cn-hangzhou.aliyuncs.com/aiminders/mpi:latest

RUN pip install gym==0.10.5

RUN mkdir /root/github
ENV CODE_DIR /root/github
# install multiagent-particle-envs
WORKDIR  $CODE_DIR
RUN cd $CODE_DIR
RUN rm -rf multiagent-particle-envs
RUN git clone https://github.com/iminders/multiagent-particle-envs.git
# Clean up pycache and pyc files
RUN cd $CODE_DIR/multiagent-particle-envs && rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install -e .

RUN pip install mpi4py
RUN pip install tensorflow==1.13.1
