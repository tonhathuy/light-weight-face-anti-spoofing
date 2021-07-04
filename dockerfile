FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /root
COPY ./ ./
RUN apt-get update && apt-get install -y wget llvm
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libpython3.7
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD nvidia-smi; sh start_service.sh	
