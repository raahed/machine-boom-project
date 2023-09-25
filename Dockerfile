FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ARG AGX_VERSION
ARG AGX_DISTRIBUTION
ENV DEBIAN_FRONTEND=noninteractive

SHELL [ "/bin/bash", "-c" ]

# Update dependency lists
RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    # Install utils apt's
    curl git tmux vim nano \
    # Install desktop
    xvfb \
    # Install connections
    openssh-server ssh x11vnc

# Install development tools
RUN apt-get install -y --no-install-recommends \
    # C and Cpp tools
    build-essential cmake \
    # Python
    python3.10 python3-pip \
    # GL stuff
    mesa-utils mesa-utils-extra glmark2 libgl1-mesa-dri libopengl-dev libgl1-mesa-dev

# Install Python tools
RUN pip3 install --upgrade pip && \
    # Baisc 
    pip3 install numpy scikit-learn pandas jupyter matplotlib opencv-python && \
    # Tensorflow
    pip3 install tensorboard tensorflow && \
    # PyTorch
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install AGX dynamics and place lic file
COPY agx-${AGX_VERSION}-${AGX_DISTRIBUTION}.deb /
RUN apt-get install -y /agx-${AGX_VERSION}-${AGX_DISTRIBUTION}.deb
COPY *.lic /opt/Algoryx/AGX-${AGX_VERSION}/
RUN source /opt/Algoryx/AGX-${AGX_VERSION}/setup_env.bash
RUN rm /agx-${AGX_VERSION}-${AGX_DISTRIBUTION}.deb

# Load agx envs every time
RUN echo "source /opt/Algoryx/AGX-${AGX_VERSION}/setup_env.bash" >> ~/.bashrc

# Enable root login in ssh
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Change passwd
RUN echo "root:default" | chpasswd

# Clean up
RUN apt-get clean -y             && \
    apt-get autoremove -y        && \
    rm -rf /var/lib/apt/lists/*


EXPOSE 5900 22 8888

ENTRYPOINT [ "/sbin/init", "-D" ]