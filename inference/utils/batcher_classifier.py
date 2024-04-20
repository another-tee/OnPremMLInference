# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import sys
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
#                               Define functions                              #
# --------------------------------------------------------------------------- #
class ClassifierImageBatcher:
    """
    Creates batches of pre-processed images.
    """

    def __init__(
            self, 
            shape, 
            dtype,
            preprocessor=None):

        # Handle Tensor Shape
        self.dtype = dtype
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Indices
        self.image_index = 0
        self.batch_index = 0
        self.preprocessor = preprocessor
    
    def preprocess_numpy_input(self, x, data_format, mode):
        """Preprocesses a Numpy array encoding a batch of images.

        Args:
            x: Input array, 3D or 4D.
            data_format: Data format of the image array.
            mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

        Returns:
            Preprocessed Numpy array.
        """

        if mode == 'tf':
            x /= 127.5
            x -= 1.
            return x
        elif mode == 'torch':
            x /= 255.
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            if data_format == 'channels_first':
                # 'RGB'->'BGR'
                if x.ndim == 3:
                    x = x[::-1, ...]
                else:
                    x = x[:, ::-1, ...]
            else:
                # 'RGB'->'BGR'
                x = x[..., ::-1]
                mean = [103.939, 116.779, 123.68]
                std = None

        # Zero-center by mean pixel
        if data_format == 'channels_first':
            if x.ndim == 3:
                x[0, :, :] -= mean[0]
                x[1, :, :] -= mean[1]
                x[2, :, :] -= mean[2]
                if std is not None:
                    x[0, :, :] /= std[0]
                    x[1, :, :] /= std[1]
                    x[2, :, :] /= std[2]
            else:
                x[:, 0, :, :] -= mean[0]
                x[:, 1, :, :] -= mean[1]
                x[:, 2, :, :] -= mean[2]
                if std is not None:
                    x[:, 0, :, :] /= std[0]
                    x[:, 1, :, :] /= std[1]
                    x[:, 2, :, :] /= std[2]
        else:
            x[..., 0] -= mean[0]
            x[..., 1] -= mean[1]
            x[..., 2] -= mean[2]
            if std is not None:
                x[..., 0] /= std[0]
                x[..., 1] /= std[1]
                x[..., 2] /= std[2]
        
        return x

    def preprocess_image(self, imagePIL):
        # Tested on inference no diff!!
        scale = None
        image = imagePIL.convert(mode='RGB')

        # Fixed shape resizer
        image = image.resize(
            (self.width, self.height), resample=Image.BILINEAR
        )

        image = np.asarray(image, dtype=self.dtype)
        if self.preprocessor in \
            ["mobilenet", "mobilenet_v2", "inception_resnet_v2", 
             "inception_v3", "nasnet", "resnet_v2", "xception"]:
            image = self.preprocess_numpy_input(image, None, "tf")
        elif self.preprocessor in ["resnet", "vgg16", "vgg19", ""]:
            image = self.preprocess_numpy_input(image, None, "caffe")
        elif self.preprocessor == "densenet":
            image = self.preprocess_numpy_input(image, None, "torch")
        elif self.preprocessor in ["mobilenet_v3", "efficientnet"]:
            pass
        else:
            print("Preprocessing: {} not supported".format(self.preprocessor))
            sys.exit(1)
        if self.format == "NCHW":
            image = np.transpose(image, (2, 0, 1))
        return image, scale

    def get_batch(self, batchesPIL):
        batch_data = np.zeros(self.shape, dtype=self.dtype)
        batch_scales = [None] * len(batchesPIL)
        
        for i, imagePIL in enumerate(batchesPIL):
            self.image_index += 1
            batch_data[i], batch_scales[i] = self.preprocess_image(imagePIL)

        self.batch_index += 1
        yield batch_data, batchesPIL, batch_scales