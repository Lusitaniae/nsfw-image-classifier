import time

from collections import defaultdict
from indexer.visual import VisualDescriptor
from indexer.celery_init import celery_app
from indexer.logger import log
from indexer import stats
from indexer.preprocessing import VideoStream
from celery.contrib import rdb

vd = VisualDescriptor()
vid = VideoStream(vd)

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
                results['response'] = result
                results['status_code'] = 2000
            elif type(image_url) is list:
                if len(image_url) <= 10:
                    list_results = []
                    for index, url in enumerate(image_url):
                        image_data = vd.url2image(thumbnail_url[index])
                        result = vd.get_tag_image(image_data)
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
    results = dict()
    results['gif_url'] = gif_url
    if gif_url:
        try:
            start = time.time()
            if type(gif_url) is str or type(gif_url) is unicode:
                # get duration of gif url
                duration = vid.get_video_duration(gif_url)
                if duration > 60:
                    results['status_code'] = 4009
                    return results

                # only extracting visual content
                results = vid.index_frames_from_url(gif_url, 1000)
                log.info("body message being indexed in all_videos: " + str())

                average_result = defaultdict(float)
                max_result = defaultdict(float)
                count = len(results)
                for result in results:
                    for key in result.keys:
                        average_result[key] = ((count - 1) * average_result[key] + result[key])/count
                        max_result[key] = max(max_result[key], result[key])

                results['response'] = average_result
                results['status_code'] = 2000
            dt = int((time.time() - start) * 1000)
            stats.timing('nsfw-gif-analytics', dt)

        except Exception as e:
            results['status_code'] = 4004
            log.error("URL is not a valid url! " + str(gif_url))
            log.info("Malformed url. Acknowledging the message...Error due to" + str(e))

    return results