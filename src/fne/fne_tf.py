import numpy as np

from fne.data_handle import DataHandle


class FeatureExtractionTensorflow:

    def __init__(self, loaded_model, image_paths, batch_size, input_tensor, target_tensors, input_reshape,
                 data_handle: DataHandle, n_crops=1, load_crops=False, reduction_factor=None, min_axis_size=None):
        self.input_reshape = input_reshape
        self.target_tensors = target_tensors
        self.input_tensor = input_tensor
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

    def _declare_empty_batch(self, n_images_in_batch, channels_first):
        if channels_first:
            img_batch = np.zeros((n_images_in_batch, 3, *self.input_reshape),
                                 dtype=np.float32)
        else:
            img_batch = np.zeros((n_images_in_batch, *self.input_reshape, 3),
                                 dtype=np.float32)
        return img_batch

    def extract_features(self):
        # Prepare output variable
        graph = self.loaded_model.graph

        # Prepare tensors to capture
        tensor_outputs = []
        for tname in self.target_tensors:
            t = graph.get_tensor_by_name(tname)
            tensor_outputs.append(t)

        len_features = 0
        for tensor_name in self.target_tensors:
            t_tensor = graph.get_tensor_by_name(tensor_name)
            len_features += t_tensor.get_shape().as_list()[-1]
        features = np.empty((len(self.image_paths) * self.number_of_crops, len_features), dtype=np.float32)
        x0 = graph.get_tensor_by_name(self.input_tensor)

        # This is the only way I found to turn a TensorShape object into a tuple
        input_tensor_shape = [x0.shape[i] for i in range(4)]
        channels_first = DataHandle.check_the_channels_first(input_tensor_shape)
        idx = 0
        for image_idx in range(0, len(self.image_paths), self.batch_size):
            batch_images_path = self.image_paths[image_idx:image_idx + self.batch_size]
            n_images_in_batch = len(batch_images_path) * self.number_of_crops
            shape, variable_input_shape = self.data_handle.get_batch_image_array_shape(batch_images_path,
                                                                                       self.input_reshape,
                                                                                       self.number_of_crops,
                                                                                       self.reduction_factor,
                                                                                       self.min_axis_size,
                                                                                       self.load_crops)
            img_batch = np.zeros((n_images_in_batch, *shape, 3), dtype=np.float32)
            #img_batch = self._declare_empty_batch(n_images_in_batch, channels_first)
            i = 0
            for img_path in batch_images_path:
                img_batch[i:i + self.number_of_crops] = self.data_handle.load_image_and_resize(img_path,
                                                                        self.input_reshape,  variable_input_shape,
                                                                        shape, self.reduction_factor,
                                                                        self.min_axis_size, channels_first,
                                                                        self.load_crops, self.number_of_crops)

                i += self.number_of_crops

            feature_vals = self.loaded_model.run(tensor_outputs, feed_dict={x0: img_batch})
            features_current = np.empty((n_images_in_batch, 0), dtype=np.float32)

            for feat in feature_vals:
                # If its not a conv layer, add without pooling
                if len(feat.shape) != 4:
                    features_current = np.concatenate((features_current, feat), axis=1)
                    continue
                # If its a conv layer, do SPATIAL AVERAGE POOLING
                pooled_vals = np.mean(np.mean(feat, axis=2, dtype=np.float32), axis=1, dtype=np.float32)
                features_current = np.concatenate((features_current, pooled_vals), axis=1)
            # Store in position
            features[idx:idx + n_images_in_batch] = features_current.copy()
            idx += n_images_in_batch
        return features
