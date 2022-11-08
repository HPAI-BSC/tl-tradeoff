import numpy as np


class PostProcessing:

    def full_network_embedding(self, features):
        features = self.standardize(features)
        features = self.discretize(features)
        return features

    @staticmethod
    def discretize(features):
        th_pos = 0.15
        th_neg = -0.25
        features[features > th_pos] = 1
        features[features < th_neg] = -1
        features[[(features >= th_neg) & (features <= th_pos)][0]] = 0

        return features

    @staticmethod
    def standardize(features):
        len_features = len(features[1])
        stats = np.empty((0, 0), dtype=np.float32)

        # Compute statistics if needed
        if len(stats) == 0:
            stats = np.zeros((2, len_features), dtype=np.float32)
            stats[0, :] = np.mean(features, axis=0, dtype=np.float32)
            stats[1, :] = np.std(features, axis=0, dtype=np.float32)

        # Apply statistics, avoiding nans after division by zero
        features = np.divide(features - stats[0], stats[1], out=np.zeros_like(features, dtype=np.float32),
                             where=stats[1] != 0)
        if len(np.argwhere(np.isnan(features))) != 0:
            raise Exception('There are nan values after standardization!')

        return features

    def l2_norm(self, features, layer_dict):
        for layer in layer_dict:
            start_layer = layer_dict[layer][0]
            end_layer = layer_dict[layer][1]
            features[:, start_layer:end_layer] = self._image_normalization_l2(features[:, start_layer:end_layer])
        return features

    @staticmethod
    def _image_normalization_l2(data_matrix):
        """Normalize the data matrix for each image
        """
        l2_norm = np.sqrt(np.sum(np.power(data_matrix, 2), axis=1))[:, np.newaxis]
        return np.nan_to_num(data_matrix / l2_norm)
