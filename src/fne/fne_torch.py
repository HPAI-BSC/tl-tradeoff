import numpy as np
import torch

from fne.data_handle import DataHandle


class FeatureExtractionTorch:

    def __init__(self, model, image_paths, batch_size, target_tensors, input_reshape, data_handle: DataHandle,
                 n_crops=1, load_crops=False, reduction_factor=0.875, min_axis_size=None):
        self.target_tensors = target_tensors
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.model = model
        self.data_handle = data_handle

        if n_crops not in [1, 10]:
            raise ValueError('The number of crops should be either 1 (no crops) or 10.')
        if n_crops == 10:
            if reduction_factor is None:
                raise ValueError('You must provide reduction factor or input to crop images.')
            if input_reshape is not None and min_axis_size is not None:
                raise ValueError('Input reshape and DownAxis (min_axis_size) are not compatible.')

        self.number_of_crops = n_crops
        if load_crops is False:
            self.reduction_factor = reduction_factor
        else:
            self.reduction_factor = None
        self.input_reshape = input_reshape
        self.min_axis_size = min_axis_size

        if load_crops is True and n_crops != 10:
            raise ValueError('For loading crops, number of crops should be 10.')
        self.load_crops = load_crops

    def extract_features(self):
        features = None
        for idx in range(0, len(self.image_paths), self.batch_size):
            batch_images_path = self.image_paths[idx:idx + self.batch_size]
            n_images_in_batch = len(batch_images_path) * self.number_of_crops

            max_resolution, variable_input_shape = self.data_handle.get_batch_image_array_shape(batch_images_path,
                                                                                                self.input_reshape,
                                                                                                self.number_of_crops,
                                                                                                self.reduction_factor,
                                                                                                self.min_axis_size,
                                                                                                self.load_crops)
            img_batch = np.zeros((n_images_in_batch, *max_resolution, 3), dtype=np.float32)
            channels_first = False
            i = 0
            for img_path in batch_images_path:
                res = self.data_handle.load_image_and_resize(img_path, self.input_reshape, variable_input_shape,
                                                             max_resolution, self.reduction_factor, self.min_axis_size,
                                                             channels_first, self.load_crops, self.number_of_crops)
                img_batch[i:i + self.number_of_crops] = np.asarray(res)
                i += self.number_of_crops

            tensor = torch.as_tensor(img_batch, dtype=torch.float32).transpose(1, 3)
            features_batch = self.model.forward(tensor)

            # Define features array shape
            if features is None:
                len_features = 0
                for target_name in self.target_tensors:
                    t_tensor = features_batch[target_name].transpose(1, 3).detach().cpu().numpy()
                    len_features += t_tensor.shape[-1]
                features = np.empty((len(self.image_paths) * self.number_of_crops, len_features), dtype=np.float32)

            n_images_in_batch = len(batch_images_path) * self.number_of_crops
            feature_vals = [features_batch[i].transpose(1, 3).detach().numpy() for i in self.target_tensors]
            features_current = np.empty((n_images_in_batch, 0), dtype=np.float32)
            for feat in feature_vals:
                pooled_vals = np.mean(np.mean(feat, axis=2, dtype=np.float32), axis=1, dtype=np.float32)
                features_current = np.concatenate((features_current, pooled_vals), axis=1)

            idx_crops = idx * self.number_of_crops
            features[idx_crops:idx_crops + n_images_in_batch, :] = features_current.copy()

        return features
