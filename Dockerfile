FROM nvidia/cuda:10.2-devel-ubuntu18.04 as base
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# add installs
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get -y upgrade
RUN apt-get purge -y python3 python3.6
RUN apt-get install -y python3.7 python3.7-dev curl python3-distutils python3-apt git ffmpeg libsm6 libxext6  -y
# update links
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.7 /usr/bin/python
RUN ln -s /usr/bin/python3.7 /usr/bin/python3
# install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
RUN python -m pip install --upgrade pip
# adding code
COPY requirements.txt .
RUN pip install -r requirements.txt
# installing apex
RUN git clone https://github.com/NVIDIA/apex /apex
RUN cd /apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN pip list

RUN cd /apex && pip install -v --disable-pip-version-check --no-cache-dir ./
RUN pip list

WORKDIR /code
COPY . .

FROM base

ENV DATASET=FALLINGTHINGS_RGB_D
ENV TEST_MODALITY=rgb
ENV EPOCHS=5
VOLUME /data
VOLUME /result

CMD python train.py --dataset=$DATASET --input_dir=/data --batch_size=16 --ndf=128 --n_rkhs=256 --n_depth=3 --output_dir=/result/ --modality=dual --modality_to_test=$TEST_MODALITY --epochs=$EPOCHS
