FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Instala dependencias básicas del sistema
RUN apt-get update && \
    apt-get install -y python3 python3-pip ffmpeg git && \
    apt-get clean

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt /workspace/
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /workspace/
RUN mkdir -p /workspace/temp

EXPOSE 5000

CMD ["python3", "app.py"]
