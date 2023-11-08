FROM ubuntu:22.04

EXPOSE 80

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libhdf5-dev libopenblas-dev protobuf-compiler libprotobuf-dev libboost-all-dev

