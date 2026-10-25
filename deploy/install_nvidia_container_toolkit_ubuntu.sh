#!/usr/bin/env bash
set -euo pipefail

# NVIDIA's supported apt installation flow for Ubuntu/Debian. The version is
# pinned so benchmark hosts remain reproducible; override deliberately when
# validating a newer release.
TOOLKIT_VERSION="${NVIDIA_CONTAINER_TOOLKIT_VERSION:-1.20.1-1}"

sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes \
      -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
sudo apt-get install -y \
  "nvidia-container-toolkit=${TOOLKIT_VERSION}" \
  "nvidia-container-toolkit-base=${TOOLKIT_VERSION}" \
  "libnvidia-container-tools=${TOOLKIT_VERSION}" \
  "libnvidia-container1=${TOOLKIT_VERSION}"

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
