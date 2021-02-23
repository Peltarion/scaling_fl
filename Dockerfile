FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime AS train

RUN apt-get update && apt-get install -y git

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD scaling_fl /src/scaling_fl
ADD scripts /src/scripts
WORKDIR /src

ARG SACRED_USER
ARG SACRED_PASSWORD
ARG SACRED_DATABASE
ARG SACRED_HOST

# Experiment tracking
ENV SACRED_USER=$SACRED_USER
ENV SACRED_PASSWORD=$SACRED_PASSWORD
ENV SACRED_DATABASE=$SACRED_DATABASE
ENV SACRED_HOST=$SACRED_HOST

# PyTorch distributed default to single node
ENV WORLD_SIZE=1
ENV RANK=0
ENV MASTER_ADDR="127.0.0.1"
ENV MASTER_PORT=23456

ENV PYTHONPATH='/src'
ENTRYPOINT ["python", "-u"]


FROM train AS debug
RUN pip install debugpy
ENTRYPOINT ["python", "-u", \
            "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client"]
