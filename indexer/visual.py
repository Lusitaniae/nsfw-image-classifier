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
import caffe
import urllib2 as urllib
import io
import imageio
import math
from indexer.logger import log
imageio.plugins.ffmpeg.download()


class VisualDescriptor():
    """
    Analyzes the image url from S3 object ID
    """

    def __init__(self, debug=False):
        pycaffe_dir = os.path.dirname(__file__)
        self.model_def = os.path.join(pycaffe_dir, 'nsfw_model/deploy.prototxt')
        self.pretrained_model = os.path.join(pycaffe_dir, 'nsfw_model/resnet_50_1by2_nsfw.caffemodel')
        # Pre-load caffe model.
        self.nsfw_net = caffe.Net(self.model_def,  # pylint: disable=invalid-name
                                  self.pretrained_model, caffe.TEST)
        # Load transformer
        # Note that the parameters are hard-coded for best results
        self.caffe_transformer = caffe.io.Transformer({'data': self.nsfw_net.blobs['data'].data.shape})
        self.caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
        self.caffe_transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        self.caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        self.caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

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

    def resize_image(self, data, sz=(256, 256)):
        """
        Resize image. Please use this resize logic for best results instead of the
        caffe, since it was used to generate training dataset
        :param str data:
            The image data
        :param sz tuple:
            The resized image dimensions
        :returns bytearray:
            A byte array with the resized image
        """
        img_data = str(data)
        im = Image.open(StringIO(img_data))
        if im.mode != "RGB":
            im = im.convert('RGB')
        imr = im.resize(sz, resample=Image.BILINEAR)
        fh_im = StringIO()
        imr.save(fh_im, format='JPEG')
        fh_im.seek(0)
        return bytearray(fh_im.read())


    def caffe_preprocess_and_compute(self, pimg, caffe_transformer=None, caffe_net=None,
                                     output_layers=None):
        """
        Run a Caffe network on an input image after preprocessing it to prepare
        it for Caffe.
        :param PIL.Image pimg:
            PIL image to be input into Caffe.
        :param caffe.Net caffe_net:
            A Caffe network with which to process pimg afrer preprocessing.
        :param list output_layers:
            A list of the names of the layers from caffe_net whose outputs are to
            to be returned.  If this is None, the default outputs for the network
            are returned.
        :return:
            Returns the requested outputs from the Caffe net.
        """
        if caffe_net is not None:

            # Grab the default output names if none were requested specifically.
            if output_layers is None:
                output_layers = caffe_net.outputs

            img_data_rs = self.resize_image(pimg, sz=(256, 256))
            image = caffe.io.load_image(StringIO(img_data_rs))

            H, W, _ = image.shape
            _, _, h, w = caffe_net.blobs['data'].data.shape
            h_off = max((H - h) / 2, 0)
            w_off = max((W - w) / 2, 0)
            crop = image[h_off:h_off + h, w_off:w_off + w, :]
            transformed_image = caffe_transformer.preprocess('data', crop)
            transformed_image.shape = (1,) + transformed_image.shape

            input_name = caffe_net.inputs[0]
            all_outputs = caffe_net.forward_all(blobs=output_layers,
                                                **{input_name: transformed_image})

            outputs = all_outputs[output_layers[0]][0].astype(float)
            return outputs
        else:
            return []

    def url2image(self, url):
        """
        Retrieves image as PIL.Image from url.
        """
        try:
            fd = urllib.urlopen(url)
            image_file = io.BytesIO(fd.read())
            image = Image.open(image_file)
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
            scores = self.caffe_preprocess_and_compute(image_data, caffe_transformer=self.caffe_transformer,
                                                       caffe_net=self.nsfw_net,
                                                       output_layers=['prob'])

            # Scores is the array containing SFW / NSFW image probabilities
            # scores[1] indicates the NSFW probability
            result['nsfw'] = scores[1]
            result['sfw'] = scores[0]
        except Exception as e:
            print("There was an error in prediction with error" + str(e))

        return result

    def index_frames_from_url(self, video_url, delay, is_extracted=False):
        if not is_extracted:
            # Extract the frames from the video and sort them for access
            results = self.index_images_from_stream(video_url, delay)
            return results
        else:
            return None

    def index_images_from_stream(self, stream_url, delay):
            """
            Sample images from a live stream at regular intervals.

            :param stream_url: url where the stream is located
            :param delay: millisecond delay between sampled images
            :return: image_batch - NxCxHxW - (num images)x(dims of image) converted w/ image_to_caffe_format
            """
            # Save images into temporary directory specific to this stream.
            fps = float(1000)/delay
            log.info("Reading from video stream and saving it as numpy array...")
            video_reader = imageio.get_reader(stream_url)
            video_info = video_reader.get_meta_data()
            video_fps = int(math.floor(video_info['fps']))
            video_duration = int(math.floor(video_info['duration']))
            if video_duration < 1.00:
                fps = float(250)/delay
            results = []
            try:
                for ind, frame in enumerate(video_reader):
                    if ind % (video_fps/fps) == 0:
                        image_data = Image.fromarray(np.uint8(frame))
                        if image_data.mode != "RGB":
                            image_data = image_data.convert('RGB')
                        fh_im = StringIO()
                        image_data.save(fh_im, format='JPEG')
                        fh_im.seek(0)
                        image_data = bytearray(fh_im.read())
                        result = self.get_tag_image(image_data)
                        results.append(result)
            except Exception as e:
                log.error(e)
                log.exception("There was an error reading a frame. Other frames available in sampled_frames.")

            return results