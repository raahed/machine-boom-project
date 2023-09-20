FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ARG AGX_VERSION
ARG AGX_DISTRIBUTION
ARG DEBIAN_FRONTEND=noninteractive

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
RUN apt-get install -y python3.10 python3-pip build-essential cmake

# Install AGX dynamics and place lic file
COPY agx-${AGX_VERSION}-${AGX_DISTRIBUTION}.deb /
RUN apt-get install -y /agx-${AGX_VERSION}-${AGX_DISTRIBUTION}.deb
COPY *.lic /opt/Algoryx/AGX-${AGX_VERSION}/
RUN source /opt/Algoryx/AGX-${AGX_VERSION}/setup_env.bash
RUN rm /agx-${AGX_VERSION}-${AGX_DISTRIBUTION}.deb

# Enable root login in ssh
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# Change passwd
RUN echo "root:default" | chpasswd

# Load agx envs every time
RUN echo "source /opt/Algoryx/AGX-${AGX_VERSION}/setup_env.bash" >> ~/.bashrc

# Clean up
RUN apt-get clean -y             && \
    apt-get autoremove -y        && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 5900 22

ENTRYPOINT [ "/sbin/init", "-D" ]