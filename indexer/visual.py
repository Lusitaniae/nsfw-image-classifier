#!/usr/bin/env python
"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license.
Please see LICENSE file in the project root for terms.
"""

import numpy as np
import os
import time
from PIL import Image
from StringIO import StringIO
import cStringIO
import caffe
import urllib2 as urllib
import io
import imageio
import math
import cv2
from indexer.logger import log
from caffe.proto import caffe_pb2
imageio.plugins.ffmpeg.download()


class VisualDescriptor():
    """
    Analyzes the image url from S3 object ID
    """

    def __init__(self, debug=False):
        pycaffe_dir = os.path.dirname(__file__)
        self.model_def = os.path.join(pycaffe_dir, 'nsfw_model/alexnet.prototxt')
        self.pretrained_model = os.path.join(pycaffe_dir, 'nsfw_model/alexnet.caffemodel')
        # Pre-load caffe model.
        self.nsfw_net = caffe.Net(self.model_def,  # pylint: disable=invalid-name
                                  self.pretrained_model, caffe.TEST)

        '''
        Reading mean image, caffe model and its weights
        '''
        # Read mean image
        self.mean_blob = caffe_pb2.BlobProto()
        with open('nsfw_model/mean.binaryproto') as f:
            self.mean_blob.ParseFromString(f.read())
        self.mean_array = np.asarray(self.mean_blob.data, dtype=np.float32).reshape(
            (self.mean_blob.channels, self.mean_blob.height, self.mean_blob.width))

        # Define image transformers
        self.caffe_transformer = caffe.io.Transformer({'data': self.nsfw_net.blobs['data'].data.shape})
        self.caffe_transformer.set_mean('data', self.mean_array)
        self.caffe_transformer.set_transpose('data', (2, 0, 1))
        self.image_width = 227
        self.image_height = 227

    def get_video_duration(self, stream_url):
        """
        Get information about video from stream url

        :param stream_url: url where the stream is located
        :return: Duration of the video
        """
        # Extract video information from stream url.
        video_reader = imageio.get_reader(stream_url)
        video_info = video_reader.get_meta_data()
        video_duration = float(video_info['duration'])

        return video_duration

    def transform_img(self, img):

        # Histogram Equalization
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        # Image Resizing
        img = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)

        return img

    def caffe_preprocess_and_compute(self, pimg):
        """
        Run a Caffe network on an input image after preprocessing it to prepare
        it for Caffe.
        :param Image numpy array:
        :param caffe.Net caffe_net:
            A Caffe network with which to process pimg afrer preprocessing.
        :param list output_layers:
            A list of the names of the layers from caffe_net whose outputs are to
            to be returned.  If this is None, the default outputs for the network
            are returned.
        :return:
            Returns the requested outputs from the Caffe net.
        """
        img = self.transform_img(pimg)
        self.nsfw_net.blobs['data'].data[...] = self.caffe_transformer.preprocess('data', img)
        out = self.nsfw_net.forward()
        outputs = out['prob']
        return outputs

    def url2image(self, url):
        """
        Retrieves image as numpy array from url.
        """
        try:
            # download the image, convert it to a NumPy array, and then read
            # it into OpenCV format
            resp = urllib.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except urllib.URLError:
            # Timed out.
            raise IOError("urllib timed out.")
        return image

    def get_tag_image(self, image_data):
        """
        Extract image descriptors
        :param image_data: PIL.Image pimg:
            PIL image to be input into Caffe.
        :return: Dictionary containing 'nsfw' and 'sfw' content
        """
        print("Predicting results for list of images...")
        result = dict()
        try:
            # Make the actual forward pass and get predictions
            scores = self.caffe_preprocess_and_compute(image_data)
            # Scores is the array containing SFW / NSFW image probabilities
            # scores[1] indicates the NSFW probability
            result['nsfw'] = scores[1]
            result['sfw'] = scores[0]
        except Exception as e:
            print("There was an error in prediction with error" + str(e))

        return result

    def index_images_from_url(self, video_url, delay):
            """
            Sample images from a live stream at regular intervals.

            :param stream_url: url where the stream is located
            :param delay: millisecond delay between sampled images
            :return: image_batch - NxCxHxW - (num images)x(dims of image) converted w/ image_to_caffe_format
            """
            # Save images into temporary directory specific to this stream.
            fps = float(1000)/delay
            log.info("Reading from video stream and saving it as numpy array...")
            video_reader = imageio.get_reader(video_url)
            video_info = video_reader.get_meta_data()
            video_fps = int(math.floor(video_info['fps']))
            video_duration = int(math.floor(video_info['duration']))
            if video_duration < 1.00:
                fps = float(250)/delay
            results = []
            try:
                for ind, frame in enumerate(video_reader):
                    if ind % (video_fps/fps) == 0:
                        image_data = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                        result = self.get_tag_image(image_data)
                        results.append(result)
            except Exception as e:
                log.error(e)
                log.exception("There was an error reading a frame. Other frames available in sampled_frames.")

            return results