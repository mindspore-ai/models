ARG FROM_IMAGE_NAME
# define base docker image
FROM $FROM_IMAGE_NAME
ARG SDK_PKG
# install packages into docker image
COPY ./$SDK_PKG . 
# install SDK
RUN ./$SDK_PKG --install
# enable env variables
RUN /bin/bash -c "source ~/.bashrc"