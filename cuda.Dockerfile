FROM nvidia/cuda:10.2-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        curl \
        build-essential \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

WORKDIR /root

ADD https://amadeus-pinnouse.s3.ca-central-1.amazonaws.com/amadeus-data.tar.gz /root
RUN tar -xf amadeus-data.tar.gz && rm amadeus-data.tar.gz

RUN pip install torch==1.7.1 torchvision==0.8.2

COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

COPY . .

CMD [ "python3", "train.py" ]