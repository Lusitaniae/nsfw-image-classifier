#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install nsfw models
if [ ! -f $DIR/nsfw_model/googlenet.caffemodel ]; then
    wget "https://s3-us-west-1.amazonaws.com/acusense-models/cnn/nsfw/googlenet.caffemodel" \
    -O $DIR/nsfw_model/googlenet.caffemodel
fi

# Install nsfw models
if [ ! -f $DIR/nsfw_model/resnet_50_1by2_nsfw.caffemodel ]; then
    wget "https://s3-us-west-1.amazonaws.com/acusense-models/cnn/nsfw/resnet_50_1by2_nsfw.caffemodel" \
    -O $DIR/nsfw_model/resnet_50_1by2_nsfw.caffemodel
fi