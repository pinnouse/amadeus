FROM pytorch/pytorch:latest

RUN apt-get update
RUN apt-get install -y build-essential

RUN curl https://amadeus-pinnouse.s3.ca-central-1.amazonaws.com/data.tar.gz | tar -xz

RUN pip install scikit-learn==0.23.2
RUN pip install pytorch-fast-transformers==0.3.0
RUN pip install performer-pytorch==0.5.0
RUN pip install reformer-pytorch==1.2.0
RUN pip install torchviz==0.0.1
RUN pip install tokenizers==0.9.3