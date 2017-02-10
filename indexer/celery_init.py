from __future__ import absolute_import
from celery import Celery
from kombu import Exchange, Queue
from indexer import CONF

celery_app = Celery('indexer', broker=CONF['redis_options']['redis_host'],
                    backend=CONF['redis_options']['redis_host'],
                    include=['indexer.tasks'])
celery_app.config_from_object('indexer.celeryconfig')
celery_app.conf.update(CELERY_DEFAULT_QUEUE='nsfw_analytics_worker')
celery_app.conf.update(CELERY_QUEUES=(Queue('nsfw_analytics_worker', Exchange('nsfw_analytics_worker'),
                                            routing_key='nsfw_analytics_worker'),))

if __name__ == '__main__':
    celery_app.start()