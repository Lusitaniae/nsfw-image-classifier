#!/usr/bin/env python
"""
Copyright 2017 Acusense Technologies, Inc.
"""

import numpy as np
import os
import caffe
import urllib2 as urllib
import imageio
import math
import cv2
import cStringIO
from indexer.logger import log
from caffe.proto import caffe_pb2
from PIL import Image
from StringIO import StringIO
imageio.plugins.ffmpeg.download()


class VisualDescriptor():
    """
    Analyzes the image url from S3 object ID
    """

    def __init__(self, debug=False):
        pycaffe_dir = os.path.dirname(__file__)

        # googlenet
        self.model_def_googlenet = os.path.join(pycaffe_dir, 'nsfw_model/googlenet.prototxt')
        self.pretrained_model_googlenet = os.path.join(pycaffe_dir, 'nsfw_model/googlenet.caffemodel')
        self.mean_path_googlenet = os.path.join(pycaffe_dir, 'nsfw_model/googlenet_mean.binaryproto')

        # resnet
        self.model_def_resnet = os.path.join(pycaffe_dir, 'nsfw_model/resnet.prototxt')
        self.pretrained_model_resnet = os.path.join(pycaffe_dir, 'nsfw_model/resnet_50_1by2_nsfw.caffemodel')

        # Pre-load caffe model.
        self.nsfw_googlenet = caffe.Net(self.model_def_googlenet,  # pylint: disable=invalid-name
                                        self.pretrained_model_googlenet, caffe.TEST)

        self.nsfw_resnet = caffe.Net(self.model_def_resnet,  # pylint: disable=invalid-name
                                     self.pretrained_model_resnet, caffe.TEST)

        '''
        Reading mean image, caffe model and its weights
        '''
        self.image_width_googlenet = 224
        self.image_height_googlenet = 224
        # Read mean image
        self.mean_blob = caffe_pb2.BlobProto()
        with open(self.mean_path_googlenet) as f:
            self.mean_blob.ParseFromString(f.read())
        self.mean_array_googlenet = np.asarray(self.mean_blob.data, dtype=np.float32).reshape((self.mean_blob.channels,
                                                                                               self.mean_blob.height,
                                                                                               self.mean_blob.width))

        # Define image transformers for googlenet
        self.caffe_transformer_googlenet = caffe.io.Transformer({'data': self.nsfw_googlenet.blobs['data'].data.shape})
        self.caffe_transformer_googlenet.set_mean('data', self.mean_array_googlenet)
        self.caffe_transformer_googlenet.set_transpose('data', (2, 0, 1))

        # Load transformer
        self.caffe_transformer_resnet = caffe.io.Transformer({'data': self.nsfw_resnet.blobs['data'].data.shape})
        self.caffe_transformer_resnet.set_transpose('data', (2, 0, 1))  # move image channels to outermost
        self.caffe_transformer_resnet.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
        self.caffe_transformer_resnet.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
        self.caffe_transformer_resnet.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR


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

    def transform_img_googlenet(self, img):

        # Histogram Equalization
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        # Image Resizing
        img = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)

        return img

    def caffe_preprocess_and_compute_by_googlenet(self, pimg):
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
        img = self.transform_img_googlenet(pimg)
        self.nsfw_googlenet.blobs['data'].data[...] = self.caffe_transformer_googlenet.preprocess('data', img)
        out = self.nsfw_googlenet.forward()
        outputs = out['prob'][0]
        return outputs

    def url2image_googlenet(self, url):
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

    def resize_image_resnet(self, data, sz=(256, 256)):
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

    def caffe_preprocess_and_compute_resnet(self, pimg, caffe_transformer=None, caffe_net=None,
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

    def url2image_resnet(self, url):
        """
        Retrieves image as PIL.Image from url.
        """
        try:
            file = cStringIO.StringIO(urllib.urlopen(url).read())
            image_data = Image.open(file)
            if image_data.mode != "RGB":
                image_data = image_data.convert('RGB')
            fh_im = StringIO()
            image_data.save(fh_im, format='JPEG')
            fh_im.seek(0)
            image = bytearray(fh_im.read())
        except urllib.URLError:
            # Timed out.
            raise IOError("urllib timed out.")
        return image

    def get_tag_image_googlenet(self, image_data):
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
            scores = self.caffe_preprocess_and_compute_by_googlenet(image_data)
            # Scores is the array containing SFW / NSFW image probabilities
            # scores[1] indicates the NSFW probability
            result['nsfw'] = float(scores[1])
            result['sfw'] = float(scores[0])
        except Exception as e:
            print("There was an error in prediction with error" + str(e))

        return result

    def get_tag_image_resnet(self, image_data):
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
            scores = self.caffe_preprocess_and_compute_resnet(image_data,
                                                              caffe_transformer=self.caffe_transformer_resnet,
                                                              caffe_net=self.nsfw_resnet,
                                                              output_layers=['prob'])

            # Scores is the array containing SFW / NSFW image probabilities
            # scores[1] indicates the NSFW probability
            result['nsfw'] = scores[1]
            result['sfw'] = scores[0]
        except Exception as e:
            log.exception("There was an error in prediction with error" + str(e))

        return result


    def index_image_from_url(self, image_url):
        """
        Sample images from a live stream at regular intervals.

        :param image_url: url of image
        """
        result = dict()
        try:
            image_data_googlenet = self.url2image_googlenet(image_url)
            image_data_resnet = self.url2image_resnet(image_url)
            result_googlenet = self.get_tag_image_googlenet(image_data_googlenet)
            result_resnet = self.get_tag_image_resnet(image_data_resnet)

            if result_resnet['nsfw'] > 0.8:
                result['nsfw'] = result_resnet['nsfw']
            elif result_googlenet['nsfw'] > 0.8:
                result['nsfw'] = result_googlenet['nsfw']
            elif result_resnet['sfw'] < 0.2:
                result['nsfw'] = result_googlenet['nsfw']
            else:
                result['nsfw'] = (0.6 * result_resnet['nsfw'] + 0.4 * result_googlenet['nsfw'])
            result['sfw'] = 1 - result['nsfw']
        except:
            log.exception("There was an error reading a image url")

        return result




    def index_frames_from_url(self, video_url, delay):
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
                    result = dict()

                    # googlenet model
                    image = cv2.cvtColor(np.array(frame), cv2.IMREAD_COLOR)
                    result_googlenet = self.get_tag_image_googlenet(image)

                    # resnet model
                    image_data = Image.fromarray(np.uint8(frame))
                    if image_data.mode != "RGB":
                        image_data = image_data.convert('RGB')
                    fh_im = StringIO()
                    image_data.save(fh_im, format='JPEG')
                    fh_im.seek(0)
                    image_data = bytearray(fh_im.read())
                    result_resnet = self.get_tag_image_resnet(image_data)
                    if result_resnet['nsfw'] > 0.8:
                        result['nsfw'] = result_resnet['nsfw']
                    elif result_googlenet['nsfw'] > 0.8:
                        result['nsfw'] = result_googlenet['nsfw']
                    elif result_resnet['sfw'] < 0.2:
                        result['nsfw'] = result_googlenet['nsfw']
                    else:
                        result['nsfw'] = (0.6*result_resnet['nsfw'] + 0.4*result_googlenet['nsfw'])
                    result['sfw'] = 1 - result['nsfw']

                    results.append(result)
        except Exception as e:
            log.error(e)
            log.exception("There was an error reading a frame. Other frames available in sampled_frames.")

        return results
