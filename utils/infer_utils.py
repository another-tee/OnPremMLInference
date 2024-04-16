# --------------------------------------------------------------------------- #
#                                   Import                                    #
# --------------------------------------------------------------------------- #
import os
import sys

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
#                               Define functions                              #
# --------------------------------------------------------------------------- #

class ImageBatcher:
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

    def preprocess_image(self, imagePIL):
        
        def resize_pad(image, pad_color=(0, 0, 0)):
            """ image: PIL image"""
            # Get characteristics.
            width, height = image.size
            width_scale = width / self.width
            height_scale = height / self.height

            # Depending on preprocessor, 
            # box scaling will be slightly different.
            if self.preprocessor == "fixed_shape_resizer":
                scale = [self.width / width, self.height / height]
                image = image.resize(
                    (self.width, self.height), resample=Image.BILINEAR
                )
                return image, scale
            elif self.preprocessor == "keep_aspect_ratio_resizer":
                scale = 1.0 / max(width_scale, height_scale)
                image = image.resize(
                    (round(width * scale), round(height * scale)), 
                    resample=Image.BILINEAR
                )
                pad = Image.new("RGB", (self.width, self.height))
                pad.paste(pad_color, [0, 0, self.width, self.height])
                pad.paste(image)
                return pad, scale
            elif self.preprocessor == "efficientdet":
                width, height = image.size
                width_scale = width / self.width
                height_scale = height / self.height
                scale = 1.0 / max(width_scale, height_scale)
                image = image.resize(
                    (round(width * scale), round(height * scale)), 
                    resample=Image.BILINEAR
                )
                pad = Image.new("RGB", (self.width, self.height))
                pad.paste(pad_color, [0, 0, self.width, self.height])
                pad.paste(image)
                return pad, scale

        scale = None
        image = imagePIL.convert(mode='RGB')
        if self.preprocessor == "fixed_shape_resizer" \
            or self.preprocessor == "keep_aspect_ratio_resizer":
            # Resize & Pad with ImageNet mean values and keep as [0,255] Norm.
            image, scale = resize_pad(image, (124, 116, 104))
            image = np.asarray(image, dtype=self.dtype)
        elif self.preprocessor == "efficientdet":
            # For EfficientNet V2: Resize & Pad with ImageNet mean values 
            # and keep as [0,255] Normalization
            image, scale = resize_pad(image, (124, 116, 104))
            image = np.asarray(image, dtype=self.dtype)
            # [0-1] Normalization, Mean subtraction 
            # and Std Dev scaling are part of the EfficientDet graph, 
            # so no need to do it during preprocessing here
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
        return batch_data, batchesPIL, batch_scales