import os
import yaml
import statsd

from indexer.config import *

path = os.path.dirname(os.path.realpath('__file__'))
path = os.path.join(path, 'indexer/config.yaml')

CONF = yaml.load(open(path))
if 'SERVER_TYPE' in os.environ:
    server_type = os.environ['SERVER_TYPE']
    if server_type == "PRODUCTION":
        CONF = CONF['production']
    elif server_type == "STAGING":
        CONF = CONF['staging']
    elif server_type == "TUNNEL":
        CONF = CONF['dev-tunnel']
    else:
        CONF = CONF['dev']
else:
    if 'dev' in CONF.keys():
        CONF = CONF['dev']