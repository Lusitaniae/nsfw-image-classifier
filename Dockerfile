FROM bvlc/caffe:cpu
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y supervisor python-opencv libopencv-dev python-numpy python-dev

ARG GIT_USERNAME=devacusense
ARG GIT_PASSWORD=Acusenseisno1!
RUN touch ~/.netrc
RUN echo "machine github.com" >> ~/.netrc
RUN echo "login $GIT_USERNAME" >> ~/.netrc
RUN echo "password $GIT_PASSWORD" >> ~/.netrc
RUN git config --global user.name "Dev Acusense"
RUN git config --global user.email dev@acusense.ai

RUN git clone https://github.com/Acusense/classifier-nsfw.git
WORKDIR /workspace/classifier-nsfw
RUN pip install -r requirements.txt
RUN bash indexer/download_models.sh

ARG GIT_COMMIT
ENV GIT_COMMIT ${GIT_COMMIT}

RUN python setup.py install
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord"]
