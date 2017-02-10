FROM
RUN apt-get update
RUN apt-get install -y git python python-dev python-distribute python-pip libpq-dev ca-certificates tesseract-ocr enchant supervisor

RUN pip install --upgrade cassandra-driver==3.5.0
RUN conda install -y opencv

ARG GIT_USERNAME=devacusense
ARG GIT_PASSWORD=Acusenseisno1!
RUN touch ~/.netrc
RUN echo "machine github.com" >> ~/.netrc
RUN echo "login $GIT_USERNAME" >> ~/.netrc
RUN echo "password $GIT_PASSWORD" >> ~/.netrc
RUN git config --global user.name "Dev Acusense"
RUN git config --global user.email dev@acusense.ai

RUN git clone https://github.com/Acusense/analytics-worker.git
WORKDIR /analytics-worker
RUN pip install -r requirements.txt

RUN cp indexer/.theanorc /root
RUN bash indexer/modelzoo/weights/download_models.sh

RUN pip install git+https://github.com/Acusense/database.git@stable#egg=acusensedb
RUN pip install git+https://github.com/Acusense/rabbit.git@stable#egg=acusensemq
RUN mkdir -p /var/log/supervisor

ARG CACHEBUST=1
ARG BRANCH=stable
RUN git pull

RUN git checkout $BRANCH
RUN pip install -r requirements.txt

RUN pip install --upgrade git+https://github.com/Acusense/database.git@$BRANCH#egg=acusensedb
RUN pip install --upgrade git+https://github.com/Acusense/rabbit.git@$BRANCH#egg=acusensemq

RUN python setup.py install
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord"]
