import os

from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
from fne.data_handle import DataHandle, downaxis_shape
import numpy as np


class DataHandlePath(DataHandle):
    def __init__(self):
        super().__init__()

    def load_image_array(self, img_path, input_reshape):
        abs_path = os.path.abspath(img_path)
        image = load_img(abs_path, target_size=input_reshape)
        image = img_to_array(image)
        return image

    def get_max_resolution(self, batch_images_path, min_axis_size=None):
        img_shapes = []
        for img_path in batch_images_path:
            img = Image.open(img_path).convert('RGB')
            img.load()
            data = np.asarray(img)
            img_shapes.append(data.shape)

        # Check if all images have same shape
        variable_input_shape = not(img_shapes.count(img_shapes[0]) == len(img_shapes))

        if min_axis_size is not None:
            for i, (height, width, channels) in enumerate(img_shapes):
                target_height, target_width = downaxis_shape(min_axis_size, height, width)
                img_shapes[i] = (target_height, target_width, channels)

        if variable_input_shape:
            # Search maximum resolution
            height = max([x[0] for x in img_shapes])
            width = max([x[1] for x in img_shapes])
        else:
            height, width, _ = img_shapes[0]

        return (height, width), variable_input_shape

