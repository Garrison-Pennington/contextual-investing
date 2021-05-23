FROM tensorflow/tensorflow:latest-gpu

ADD . /root/contextual-investing
WORKDIR /root/contextual-investing
RUN mkdir -p /root/.models