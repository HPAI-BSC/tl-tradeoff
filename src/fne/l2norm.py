import os

import numpy as np
from tensorflow.python.keras.models import Model

from fne.data_handle import DataHandle
from fne.data_handle_path import DataHandlePath


class L2Norm:
    def __init__(self, loaded_model, image_paths, batch_size, target_tensors, input_reshape, data_handle:DataHandle):
        self.data_handle = data_handle
        self.input_reshape = input_reshape
        self.target_tensors = target_tensors
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.loaded_model = loaded_model

    def extract_features(self):

        for t_idx, tensor_name in enumerate(self.target_tensors):
            model = Model(inputs=self.loaded_model.input, outputs=self.loaded_model.get_layer(tensor_name).output)
            for idx in range(0, len(self.image_paths), self.batch_size):
                batch_images_path = self.image_paths[idx:idx + self.batch_size]
                img_batch = np.zeros((len(batch_images_path), *self.input_reshape, 3), dtype=np.float32)
                for i, img_path in enumerate(batch_images_path):
                    img_batch[i] = self.data_handle.load_image_and_resize(img_path, self.input_reshape)

                features_batch = model.predict(img_batch, batch_size=self.batch_size)

                # If its a conv layer, do SPATIAL AVERAGE POOLING
                if len(features_batch.shape) == 4:
                    features_batch = np.mean(np.mean(features_batch, axis=2), axis=1)
                if idx == 0:
                    features_layer = features_batch.copy()
                else:
                    features_layer = np.concatenate((features_layer, features_batch.copy()), axis=0)
                features_layer = self._image_normalization_L2(features_layer)

            if t_idx == 0:
                features = features_layer.copy()
            else:
                features = np.concatenate((features, features_layer.copy()), axis=1)

        return features

    @staticmethod
    def _image_normalization_L2(data_matrix):
        """Normalize the data matrix for each image
        """
        l2_norm = np.sqrt(np.sum(np.power(data_matrix, 2), axis=1))[:, np.newaxis]
        return np.nan_to_num(data_matrix / l2_norm)
