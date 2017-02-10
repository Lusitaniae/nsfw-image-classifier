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
def analytics_image_worker(self, image_url, thumbnail_url):
    results = dict()
    results['image_url'] = image_url
    if image_url:
        try:
            start = time.time()
            if type(image_url) is str or type(image_url) is unicode:
                image = vd.url2img(thumbnail_url)

                # tags extracted from our own models
                vd_words, vd_probs = vd.get_tags_frames([image])
                vd_words, vd_probs = vd_words[0], vd_probs[0]

                dt = int((time.time() - start) * 1000)
                stats.timing('image-analytics', dt)

                results['visual_words'] = vd_words
                results['visual_probs'] = vd_probs
                results['status_code'] = 2000
            elif type(image_url) is list:
                if len(image_url) <= 10:
                    list_vd_words, list_vd_probs = [], []
                    for index, url in enumerate(image_url):
                        image = vd.url2img(thumbnail_url[index])
                        vd_words, vd_probs = vd.get_tags_frames([image])
                        vd_words, vd_probs = vd_words[0], vd_probs[0]
                        list_vd_words.append(vd_words)
                        list_vd_probs.append(vd_probs)

                    results['visual_words'] = list_vd_words
                    results['visual_probs'] = list_vd_probs
                    results['status_code'] = 2000
                else:
                    log.info("Exceeded the limit for number of images")
                    results['status_code'] = 4001
                    return results
        except Exception as e:
            log.error("URL is not a valid url! " + str(image_url))
            log.info("Malformed url. Acknowledging the message...Error due to" + str(e))

    return results


@celery_app.task(bind=True, soft_time_limit=1000)
def analytics_gif_worker(self, gif_url):
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
                vd_words, vd_probs, timestamp_vision = vid.index_frames_from_url(gif_url, 1000)

                log.info("body message being indexed in all_videos: " + str())

                vd_words_probs = defaultdict(float)
                vd_words_count = defaultdict(float)
                all_vd_words = set()
                for index_frame, vd_word in enumerate(vd_words):
                    all_vd_words = all_vd_words.union(set(vd_word))
                    for index_word, word in enumerate(vd_word):
                        vd_words_probs[word] += vd_probs[index_frame][index_word]
                        vd_words_count[word] += 1

                all_vd_words = list(all_vd_words)
                vd_probs_avg = []
                for word in all_vd_words:
                    vd_probs_avg.append(float(vd_words_probs[word]/vd_words_count[word]))

                probs_words = zip(vd_probs_avg, all_vd_words)
                probs_words.sort(reverse=True)
                results['visual_words'] = [words for probs, words in probs_words]
                results['visual_probs'] = [probs for probs, words in probs_words]
                results['status_code'] = 2000

                dt = int((time.time() - start) * 1000)
                stats.timing('gif-analytics', dt)

        except Exception as e:
            results['status_code'] = 4004
            log.error("URL is not a valid url! " + str(gif_url))
            log.info("Malformed url. Acknowledging the message...Error due to" + str(e))

    return results