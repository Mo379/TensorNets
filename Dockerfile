#seed
FROM python:3.8-bullseye
#setup
ENV TZ=Europe/Minsk
ENV DEBIAN_FRONTEND=noninteractive 
#Python
RUN apt-get update
RUN apt update 
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get -y install apt-utils vim curl git tar
RUN pip install --upgrade pip
#Add installation files
WORKDIR /workspace/APP
#ADD ./APP/ /workspace/APP
ADD ./extras/shell/ /workspace/shell
ADD ./extras/includes/ /workspace/includes
ADD ./extras/lib/ /workspace/lib
ADD ./extras/tests/ /workspace/tests
#shell script permissions
RUN chmod +x /workspace/shell/*
RUN chmod +x /workspace/tests/*
#Run shell scripts
RUN /workspace/shell/requirements.sh
RUN /workspace/shell/jax.sh
# Tests
RUN python3 /workspace/tests/test_jax.py


#expose and run
CMD ["python3", "-V"]
