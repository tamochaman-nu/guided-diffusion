# ホストのドライバに合わせて切り替える:
#   >= 550: cuda12.4 / >= 530: cuda12.1 / >= 520: cuda11.8
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    openmpi-bin libopenmpi-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY setup.py /app/

RUN pip install --no-cache-dir "blobfile>=1.0.5" torch tqdm mpi4py Pillow

CMD ["/bin/bash"]
