# Use the official base Ubuntu image
FROM mcr.microsoft.com/devcontainers/base:ubuntu

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN apt-get update && \
    apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# Set the remote user
USER vscode
