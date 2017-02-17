FROM bvlc/caffe:cpu
RUN apt-get update
RUN apt-get install -y supervisor

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

ARG BRANCH=stable
RUN git pull

RUN git checkout $BRANCH
RUN pip install -r requirements.txt

ARG GIT_COMMIT
ENV GIT_COMMIT ${GIT_COMMIT}

RUN python setup.py install
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord"]
