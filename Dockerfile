FROM ubuntu:16.04

RUN apt -y update &&\
    apt -y install python3 python3-pip

RUN python3 -m pip install --upgrade pip

COPY . /app
WORKDIR /app

ADD ./requirements.txt /
RUN python3 -m pip install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["server.py"]