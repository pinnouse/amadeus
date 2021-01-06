FROM pytorch/pytorch:latest

RUN apt-get -q update
RUN apt-get -qy install bash g\+\+

ADD https://amadeus-pinnouse.s3.ca-central-1.amazonaws.com/amadeus-data.tar.gz /workspace/
RUN tar -xf amadeus-data.tar.gz && rm amadeus-data.tar.gz

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt