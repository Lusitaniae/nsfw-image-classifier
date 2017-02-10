import logging

sh = logging.StreamHandler()
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(fmt)

log = logging.getLogger('nsfw-analytics-worker')
log.addHandler(sh)