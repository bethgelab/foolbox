from __future__ import absolute_import

import numpy as np
import warnings
from .base import DifferentiableModel
from .. import utils


class CaffeModel(DifferentiableModel):
    def __init__(self,
                 net,
                 bounds,
                 channel_axis=1,
                 preprocessing=(0, 1),
                 data_blob_name="data",
                 label_blob_name="label",
                 output_blob_name="output"):
        super(CaffeModel, self).__init__(bounds=bounds,
                                         channel_axis=channel_axis,
                                         preprocessing=preprocessing)

        warnings.warn('Caffe was superseeded by Caffe2 and now PyTorch 1.0,'
                      ' thus Caffe support in Foolbox will be removed',
                      DeprecationWarning)

        import caffe
        self.net = net
        assert isinstance(net, caffe.Net)
        assert data_blob_name in self.net.blobs
        assert label_blob_name in self.net.blobs
        self.data_blob_name = data_blob_name
        self.label_blob_name = label_blob_name
        self.output_blob_name = output_blob_name

    def num_classes(self):
        return self.net.blobs[self.output_blob_name].data.shape[-1]

    def batch_predictions(self, images):
        images, _ = self._process_input(images)
        self.net.blobs[self.data_blob_name].reshape(*images.shape)
        self.net.blobs[self.label_blob_name].reshape(images.shape[0])
        self.net.blobs[self.data_blob_name].data[:] = images
        self.net.forward()
        return self.net.blobs[self.output_blob_name].data

    def predictions_and_gradient(self, image, label):
        input_shape = image.shape

        image, dpdx = self._process_input(image)
        self.net.blobs[self.data_blob_name].data[0, :] = image
        self.net.blobs[self.label_blob_name].data[0] = label

        self.net.forward()
        predictions = self.net.blobs[self.output_blob_name].data[0]

        grad_data = self.net.backward(diffs=[self.data_blob_name])
        grad = grad_data[self.data_blob_name][0]
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return predictions, grad

    def batch_gradients(self, images, labels):
        if images.shape[0] == labels.shape[0] == 1:
            _, g = self.predictions_and_gradient(images[0], labels[0])
            return g[np.newaxis]
        raise NotImplementedError

    def _loss_fn(self, image, label):
        logits = self.batch_predictions(image[None])
        return utils.batch_crossentropy([label], logits)

    def backward(self, gradient, image):
        input_shape = image.shape
        image, dpdx = self._process_input(image)
        self.net.blobs[self.data_blob_name].data[:] = image
        self.net.forward()
        self.net.blobs[self.output_blob_name].diff[...] = gradient
        grad_data = self.net.backward(start=self.output_blob_name,
                                      diffs=[self.data_blob_name])
        grad = grad_data[self.data_blob_name][0]
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return grad

    def batch_backward(self, gradients, images):
        if images.shape[0] == gradients.shape[0] == 1:
            return self.backward(gradients[0], images[0])[np.newaxis]
        raise NotImplementedError
