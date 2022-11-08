import os
from abc import abstractmethod, ABCMeta

import numpy as np


class FNE:
    __metaclass__ = ABCMeta

    @abstractmethod
    def extract_features(self):
        pass

    def full_network_embedding(self):
        features = self.extract_features()
        features, stats = self.standardize(features)
        features = self.discretize(features)

        # Store output
        outputs_path = '../outputs'
        if not os.path.exists(outputs_path):
            os.makedirs(outputs_path)
        np.save(os.path.join(outputs_path, 'fne.npy'), features)
        np.save(os.path.join(outputs_path, 'stats.npy'), stats)

        # Load output
        # fne = np.load('fne.npy')
        # fne_stats = np.load('stats.npy')

        # Return
        return features, stats

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

        return features, stats
