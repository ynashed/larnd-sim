FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

MAINTAINER Youssef Nashed "ynashed@slac.stanford.edu"

ENV PYTHONUNBUFFERED=1

ARG SCRATCH_VOLUME=/scratch
ENV SCRATCH_VOLUME=/scratch
RUN echo creating ${SCRATCH_VOLUME} && mkdir -p ${SCRATCH_VOLUME}
VOLUME ${SCRATCH_VOLUME}

WORKDIR /work
ADD requirements.txt /work/requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git wget build-essential

RUN mkdir -p /tmp && wget -q --no-check-certificate -P /tmp https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.4.tar.bz2 && \
    tar -x -f /tmp/openmpi-4.0.4.tar.bz2 -C /tmp -j && \
    cd /tmp/openmpi-4.0.4 && ./configure --prefix=/usr/local/openmpi --disable-getpwuid \
    --with-slurm --with-cuda && \
    make -j4 && \
    make -j4 install && \
    rm -rf /tmp/openmpi-4.0.4.tar.bz2 /tmp/openmpi-4.0.4
ENV PATH=/usr/local/openmpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ADD . /work

# ensure the script is executable
#RUN chmod +x scripts/run_module.sh
#
#ENTRYPOINT ["/bin/bash", "./scripts/run_module.sh"]
