FROM ubuntu:20.04
EXPOSE 8000
EXPOSE 6006
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget gnupg
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && apt-get install -y intel-oneapi-mkl-devel-2023.1.0
RUN apt-get update && apt-get install -y cmake build-essential git libhdf5-dev libboost-all-dev protobuf-compiler libprotobuf-dev python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorboard==2.12.2 six
RUN mkdir $HOME/.ssh && ssh-keyscan github.com >> $HOME/.ssh/known_hosts
RUN git clone https://github.com/arplaboratory/learning_to_fly.git && echo 1
WORKDIR /learning_to_fly
RUN git submodule update --init -- external/rl_tools
RUN cd src/ui && ./get_dependencies.sh
WORKDIR /learning_to_fly/external/rl_tools
RUN git submodule update --init -- external/cli11 external/highfive external/json/ external/tensorboard tests/lib/googletest/
WORKDIR /learning_to_fly
WORKDIR /
RUN mkdir build
WORKDIR /build
RUN cmake ../learning_to_fly -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON
RUN cmake --build . -j$(nproc)

WORKDIR /learning_to_fly
RUN echo "#!/bin/bash" > /init.sh && \
    chmod +x /init.sh && \
    echo "tensorboard --logdir=logs --bind_all&" >> /init.sh && \
    echo "echo Waiting for Tensorboard" >> /init.sh && \
    echo "sleep 5" >> /init.sh && \
    echo "echo Running command:" >> /init.sh && \
    echo "export PATH=\"\$PATH:/build/src\"" >> /init.sh && \
    echo "\$@" >> /init.sh
CMD ["/build/src/ui", "0.0.0.0", "8000"]
ENTRYPOINT ["/init.sh"]

