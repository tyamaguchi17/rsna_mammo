FROM nvcr.io/nvidia/pytorch:22.02-py3
ARG PYTORCH="1.11.0"
ARG TORCHVISION="0.12.0"
ARG CUDA="113"

ADD nvidia-driver.sh  /tmp/nvidia-driver.sh
# Avoiding user interaction with libopencv-dev
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y && apt install -y \
    git
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    vim \
    libopencv-dev \
    libgl1-mesa-dev \
    tmux &&\
    bash /tmp/nvidia-driver.sh &&\
    apt-get clean && \
    rm -rf /var/lib/apt/list

WORKDIR /home/working

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US
