ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

RUN apt install libgeos-dev -y
COPY requirements.txt .
RUN pip3.7 install -r requirements.txt