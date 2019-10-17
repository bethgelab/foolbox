import numpy as np
import warnings
from .base import DifferentiableModel
from .. import utils


class CaffeModel(DifferentiableModel):  # pragma: no cover
    def __init__(
        self,
        net,
        bounds,
        channel_axis=1,
        preprocessing=(0, 1),
        data_blob_name="data",
        label_blob_name="label",
        output_blob_name="output",
    ):
        super(CaffeModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocessing=preprocessing
        )

        warnings.warn(
            "Caffe was superseeded by Caffe2 and now PyTorch 1.0,"
            " thus Caffe support in Foolbox will be removed",
            DeprecationWarning,
        )

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

    def forward(self, inputs):
        inputs, _ = self._process_input(inputs)
        self.net.blobs[self.data_blob_name].reshape(*inputs.shape)
        self.net.blobs[self.label_blob_name].reshape(inputs.shape[0])
        self.net.blobs[self.data_blob_name].data[:] = inputs
        self.net.forward()
        return self.net.blobs[self.output_blob_name].data

    def forward_and_gradient_one(self, x, label):
        input_shape = x.shape

        x, dpdx = self._process_input(x)
        self.net.blobs[self.data_blob_name].data[0, :] = x
        self.net.blobs[self.label_blob_name].data[0] = label

        self.net.forward()
        predictions = self.net.blobs[self.output_blob_name].data[0]

        grad_data = self.net.backward(diffs=[self.data_blob_name])
        grad = grad_data[self.data_blob_name][0]
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return predictions, grad

    def forward_and_gradient(self, inputs, labels):
        inputs_shape = inputs.shape

        inputs, dpdx = self._process_input(inputs)
        self.net.blobs[self.data_blob_name].data[:] = inputs
        self.net.blobs[self.label_blob_name].data[:] = labels

        self.net.forward()
        predictions = self.net.blobs[self.output_blob_name].data

        grad_data = self.net.backward(diffs=[self.data_blob_name])
        grad = grad_data[self.data_blob_name]
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == inputs_shape

        return predictions, grad

    def gradient(self, inputs, labels):
        if inputs.shape[0] == labels.shape[0] == 1:
            _, g = self.forward_and_gradient_one(inputs[0], labels[0])
            return g[np.newaxis]
        raise NotImplementedError

    def _loss_fn(self, x, label):
        label = np.array(label)

        if len(label.shape) == 0:
            # add batch dimension
            label = label[np.newaxis]
            x = x[np.newaxis]

        logits = self.forward(x)
        return utils.batch_crossentropy(label, logits)

    def _backward_one(self, gradient, x):
        input_shape = x.shape
        x, dpdx = self._process_input(x)
        self.net.blobs[self.data_blob_name].data[:] = x
        self.net.forward()
        self.net.blobs[self.output_blob_name].diff[...] = gradient
        grad_data = self.net.backward(
            start=self.output_blob_name, diffs=[self.data_blob_name]
        )
        grad = grad_data[self.data_blob_name][0]
        grad = self._process_gradient(dpdx, grad)
        assert grad.shape == input_shape

        return grad

    def backward(self, gradient, inputs):
        if inputs.shape[0] == gradient.shape[0] == 1:
            return self._backward_one(gradient[0], inputs[0])[np.newaxis]
        raise NotImplementedError
