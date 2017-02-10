import json
import uuid
import pytest
import datetime
from celery.result import AsyncResult
from indexer.celery_init import celery_app
from indexer.tasks import analytics_image_worker, analytics_gif_worker
from indexer.logger import log

TEST_TYPE='celer'


def test_analytics_nsfw_image_worker():
    image_url = 'http://cdn.inspire-confidence.psdops.com/ec/57/0be1a81343a9af9b760665f39318/kilimanjaro-3.jpg'
    thumbnail_url = 'http://cdn.inspire-confidence.psdops.com/ec/57/0be1a81343a9af9b760665f39318/kilimanjaro-3.jpg'

    if TEST_TYPE == 'celery':
        args = [image_url, thumbnail_url]
        task = celery_app.send_task('indexer.tasks.analytics_nsfw_image_worker', args=args)
        result = task.wait()
    else:
        result = analytics_image_worker(image_url, thumbnail_url)

    log.info('results' + str(result))
    assert result['status_code'] == 2000


def test_analytics_nsfw_gif_worker():

    gif_url = 'https://video.twimg.com/tweet_video/C3SqA2UXUAA149x.mp4'

    if TEST_TYPE == 'celery':
        args = [gif_url]
        task = celery_app.send_task('indexer.tasks.analytics_nsfw_gif_worker', args=args)
        result = task.wait()
    else:
        result = analytics_gif_worker(gif_url)

    log.info('results' + str(result))
    assert result['status_code'] == 2000