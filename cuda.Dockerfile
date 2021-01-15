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

RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg
ENV GCLOUD_ENABLE true
RUN mkdir /tmp -p

COPY . .

CMD [ "python3", "train.py" ]