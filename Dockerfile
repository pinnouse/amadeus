FROM pytorch/pytorch:latest

RUN apt-get -q update
RUN apt-get -qy install g\+\+

ADD https://amadeus-pinnouse.s3.ca-central-1.amazonaws.com/data.tar.gz /workspace/

RUN pip install scikit-learn==0.23.2
RUN pip install performer-pytorch==0.7.4
RUN pip install tokenizers==0.9.3
RUN pip install gradient-statsd==1.0.1