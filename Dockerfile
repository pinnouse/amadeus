FROM pytorch/pytorch:latest

RUN apt update
RUN apt install -y build-essential

RUN pip install pytorch-fast-transformers==0.3.0
RUN pip install performer-pytorch==0.5.0
RUN pip install torchviz==0.0.1
RUN pip install tokenizers==0.9.3