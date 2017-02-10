import time

from collections import defaultdict
from indexer.visual import VisualDescriptor
from indexer.celery_init import celery_app
from indexer.logger import log
from indexer import stats
from celery.contrib import rdb

vd = VisualDescriptor()

@celery_app.task(bind=True, soft_time_limit=1000)
def nsfw_analytics_image_worker(self, image_url, thumbnail_url):
    results = dict()
    results['image_url'] = image_url
    if image_url:
        try:
            start = time.time()
            if type(image_url) is str or type(image_url) is unicode:
                image_data = vd.url2image(thumbnail_url)
                result = vd.get_tag_image(image_data)
                # need to remove
                result['nsfw'] *= 0.98
                result['sfw'] = 1 - result['nsfw']
                results['response'] = result
                results['status_code'] = 2000
            elif type(image_url) is list:
                if len(image_url) <= 10:
                    list_results = []
                    for index, url in enumerate(image_url):
                        image_data = vd.url2image(thumbnail_url[index])
                        result = vd.get_tag_image(image_data)
                        # need to remove
                        result['nsfw'] *= 0.98
                        result['sfw'] = 1 - result['nsfw']
                        list_results.append(result)

                    results['response'] = list_results
                    results['status_code'] = 2000
                else:
                    log.info("Exceeded the limit for number of images")
                    results['status_code'] = 4001
                    return results
            dt = int((time.time() - start) * 1000)
            stats.timing('nsfw-gif-analytics', dt)
        except Exception as e:
            log.error("URL is not a valid url! " + str(image_url))
            log.info("Malformed url. Acknowledging the message...Error due to" + str(e))

    return results


@celery_app.task(bind=True, soft_time_limit=1000)
def nsfw_analytics_gif_worker(self, gif_url):
    response = dict()
    response['gif_url'] = gif_url
    if gif_url:
        try:
            start = time.time()
            if type(gif_url) is str or type(gif_url) is unicode:
                # get duration of gif url
                duration = vd.get_video_duration(gif_url)
                if duration > 60:
                    response['status_code'] = 4009
                    return response

                # only extracting visual content
                results = vd.index_frames_from_url(gif_url, 1000)
                log.info("body message being indexed in all_videos: " + str())

                average_result = defaultdict(float)
                max_result = defaultdict(float)
                n = 0
                for result in results:
                    n += 1
                    for key in result.keys():
                        average_result[key] = ((n - 1) * average_result[key] + result[key])/n
                        max_result[key] = max(max_result[key], result[key])

                # need to remove
                max_result['nsfw'] *= 0.98
                max_result['sfw'] = 1 - max_result['nsfw']

                response['response'] = max_result
                response['status_code'] = 2000
            dt = int((time.time() - start) * 1000)
            stats.timing('nsfw-gif-analytics', dt)

        except Exception as e:
            response['status_code'] = 4004
            log.error("URL is not a valid url! " + str(gif_url))
            log.info("Malformed url. Acknowledging the message...Error due to" + str(e))

    return response