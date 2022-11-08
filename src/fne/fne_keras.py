import numpy as np
from tensorflow.python.keras.models import Model
from fne.data_handle import DataHandle


class FeatureExtractionKeras:

    def __init__(self, loaded_model, image_paths, batch_size, target_tensors, input_reshape, data_handle: DataHandle,
                 n_crops=1, load_crops=False, reduction_factor=None, min_axis_size=None):
        self.input_reshape = input_reshape
        self.target_tensors = target_tensors
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.loaded_model = loaded_model
        self.data_handle = data_handle
        self.reduction_factor = reduction_factor
        self.min_axis_size = min_axis_size

        if n_crops not in [1, 10]:
            raise ValueError('The number of crops should be either 1 (no crops) or 10.')
        self.number_of_crops = n_crops

        if load_crops is True and n_crops != 10:
            raise ValueError('For loading crops, number of crops should be 10.')
        self.load_crops = load_crops

    def extract_features(self):
        for t_idx, tensor_name in enumerate(self.target_tensors):

            model = Model(inputs=self.loaded_model.input, outputs=self.loaded_model.get_layer(tensor_name).output)
            channels_first = False

            for idx in range(0, len(self.image_paths), self.batch_size):
                batch_images_path = self.image_paths[idx:idx + self.batch_size]
                n_images_in_batch = len(batch_images_path) * self.number_of_crops

                # See if input shape is variable
                # Declare batch array with size
                max_resolution, variable_input_shape = self.data_handle.get_batch_image_array_shape(batch_images_path,
                                                                        self.input_reshape, self.number_of_crops,
                                                                        self.reduction_factor, self.min_axis_size,
                                                                        self.load_crops)
                img_batch = np.zeros((n_images_in_batch, *max_resolution, 3), dtype=np.float32)

                i = 0
                for img_path in batch_images_path:
                    res = self.data_handle.load_image_and_resize(img_path, self.input_reshape,  variable_input_shape,
                                                                        max_resolution, self.reduction_factor,
                                                                        self.min_axis_size, channels_first,
                                                                        self.load_crops, self.number_of_crops)
                    img_batch[i:i + self.number_of_crops] = np.asarray(res)
                    i += self.number_of_crops
                try:
                    features_batch = model.predict(img_batch, batch_size=self.batch_size)
                except ValueError:
                    raise ValueError("The model expected a different input shape than the one provided.")

                # If its a conv layer, do SPATIAL AVERAGE POOLING
                if len(features_batch.shape) == 4:
                    features_batch = np.mean(np.mean(features_batch, axis=2), axis=1)
                if idx == 0:
                    features_layer = features_batch.copy()
                else:
                    features_layer = np.concatenate((features_layer, features_batch.copy()), axis=0)

            if t_idx == 0:
                features = features_layer.copy()
            else:
                features = np.concatenate((features, features_layer.copy()), axis=1)

        return features
