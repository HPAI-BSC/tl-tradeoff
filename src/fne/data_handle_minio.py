from io import BytesIO

import PIL
import boto3 as boto3
import numpy as np
from PIL import Image
from botocore.config import Config

from fne.data_handle import DataHandle, downaxis_shape


class MinioClient:
    def __init__(self):
        pass

    def connect(self, endpoint_url):
        resource = boto3.resource('s3',
                                  endpoint_url=endpoint_url,
                                  config=Config(signature_version='s3v4'),
                                  region_name='us-east-1')
        return resource.meta.client

    def download(self, connection, bucket_name, file_path):
        return connection.get_object(Bucket=bucket_name,
                                     Key=file_path)


class DataHandleMinio(DataHandle):
    def __init__(self, minio_client: MinioClient, endpoint_url, bucket_name):
        super().__init__()
        self.bucket_name = bucket_name
        self.minio_client = minio_client
        self.connection = minio_client.connect(endpoint_url)

    def load_image_array(self, file_path, input_reshape):
        obj = self.minio_client.download(self.connection, self.bucket_name, file_path)
        img = Image.open(BytesIO(obj['Body'].read()))
        img_resized = img.resize(input_reshape, resample=PIL.Image.NEAREST)
        img_rgb = img_resized.convert(mode="RGB")
        np_img = np.asarray(img_rgb)

        return np_img

    def get_max_resolution(self, batch_images_path, min_axis_size=None):
        img_shapes = []
        for img_path in batch_images_path:
            obj = self.minio_client.download(self.connection, self.bucket_name, img_path)
            img = Image.open(BytesIO(obj['Body'].read()))
            img_rgb = img.convert(mode="RGB")
            data = np.asarray(img_rgb)
            img_shapes.append(data.shape)

        # Check if all images have same shape
        variable_input_shape = not (img_shapes.count(img_shapes[0]) == len(img_shapes))

        if min_axis_size is not None:
            for i, (height, width) in enumerate(img_shapes):
                target_height, target_width = downaxis_shape(min_axis_size, height, width)
                img_shapes[i] = (target_height, target_width)

        if not variable_input_shape:
            # Search maximum resolution
            height = max([x[0] for x in img_shapes])
            width = max([x[1] for x in img_shapes])
            max_resolution = (height, width)
        else:
            max_resolution = img_shapes[0]

        return max_resolution, variable_input_shape
