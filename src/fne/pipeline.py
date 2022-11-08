from time import time

import numpy as np
import tensorflow as tf
import torch
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from hpai_utils.accuracy import accuracy, accuracy_average_per_class_voting
from fne.data_handle import DataHandle
from fne.fne_tf import FeatureExtractionTensorflow
from fne.fne_keras import FeatureExtractionKeras
from fne.fne_torch import FeatureExtractionTorch
from fne.loaded_model import LoadedModelFromSession
from fne.model import ModelHandle
from fne.post_processing import PostProcessing
from fne.utils import encode_labels


class Pipeline:
    def __init__(self, model_handle: ModelHandle,
                 data_handle: DataHandle, batch_size=10, n_crops=1, load_crops=False,
                 reduction_factor=None, min_axis_size=None):
        # Handles
        self.model_handle = model_handle
        self.data_handle = data_handle

        self.batch_size = batch_size

        # model parameters
        self.model_description = self._load_model_description()
        self.image_resize = tuple(self.model_description['input_reshape'])
        self.input_tensor = self.model_description['input_tensor']
        self.target_tensors = self.model_description['target_tensors']

        if len(self.image_resize) == 0:
            self.image_resize = None

        self.layer_positions = {}
        self.n_crops = n_crops
        self.load_crops = load_crops
        self.reduction_factor = reduction_factor
        self.min_axis_size = min_axis_size

    def _load_model_description(self):
        return self.model_handle.load_model_description()

    def _load_model(self):
        self.model_handle.load_model()

    def calculate_layer_position(self, model):
        layer_positions = {}
        graph = model.graph
        layer_begin_pos = 0
        for i, tensor_name in enumerate(self.target_tensors):
            t_tensor = graph.get_tensor_by_name(tensor_name)
            len_features = t_tensor.get_shape().as_list()[-1]
            layer_end_pos = layer_begin_pos + len_features
            layer_positions[tensor_name] = (layer_begin_pos, layer_end_pos)
            layer_begin_pos = layer_end_pos
        return layer_positions

    def calculate_layer_position_torch(self, model):
        # Using fake input to get the shape since it seems the easiest way
        fake_input = torch.rand((1, 3, 224, 224))
        fake_features = model.forward(fake_input)
        len_features = 0
        layer_positions = {}
        for target_name in self.target_tensors:
            layer_end_pos = len_features + fake_features[target_name].detach().cpu().shape[1]
            layer_positions[target_name] = (len_features, layer_end_pos)
            len_features = layer_end_pos
        return layer_positions

    def extract_features(self, images, fne):
        features = None
        if fne == 'tf' or fne == 'keras':
            tf.compat.v1.reset_default_graph()
            with tf.compat.v1.Session() as sess:
                self._load_model()
                model = LoadedModelFromSession(sess)
                if not len(self.layer_positions):
                    self.layer_positions = self.calculate_layer_position(model)
                tf_fe = FeatureExtractionTensorflow(model, images, self.batch_size, self.input_tensor,
                                                    self.target_tensors, self.image_resize, self.data_handle,
                                                    self.n_crops, self.load_crops, self.reduction_factor,
                                                    self.min_axis_size)
                features = tf_fe.extract_features()
            tf.compat.v1.reset_default_graph()
        elif fne == 'torch':
            model = self.model_handle.load_model()
            self.layer_positions = self.calculate_layer_position_torch(model)
            torch_fe = FeatureExtractionTorch(model, images, self.batch_size, self.target_tensors, self.image_resize,
                                              self.data_handle, self.n_crops, self.load_crops, self.reduction_factor,
                                              self.min_axis_size)
            features = torch_fe.extract_features()
        return features

    @staticmethod
    def train_with_features(clf, features_train, train_labels, features_test):
        clf.fit(X=features_train, y=train_labels)
        predicted_labels = clf.predict(features_test)
        return predicted_labels

    @staticmethod
    def train_svm_with_features(features_train, train_labels, features_test):
        clf = LinearSVC()
        clf.fit(X=features_train, y=train_labels)
        predicted_labels = clf.predict(features_test)
        return predicted_labels

    @staticmethod
    def train_lda_with_features(features_train, train_labels, features_test):
        clf = LinearDiscriminantAnalysis()
        clf.fit(X=features_train, y=train_labels)
        predicted_labels = clf.predict(features_test)
        return predicted_labels

    def generate_configurations(self, layer_combinations):
        max_size_subset = len(self.target_tensors) + 1
        combinations = []
        if layer_combinations == 'all_contiguous':
            for i in range(max_size_subset):
                for j in range(i, max_size_subset):
                    if len(self.target_tensors[i:j]):
                        combinations.append(self.target_tensors[i:j])
        elif layer_combinations == 'full':
            combinations.append(self.target_tensors)
        elif 'last_' in layer_combinations:
            n_layers = int(layer_combinations[5:])
            combinations.append(self.target_tensors[-n_layers:])
        elif layer_combinations == 'last':
            combinations.append(self.target_tensors[-1:])
        return combinations

    def execute_postprocessings(self, features, include_l2=True):
        fne_features = PostProcessing().full_network_embedding(features)
        l2_features = PostProcessing().l2_norm(features, self.layer_positions) if include_l2 else None
        return fne_features, l2_features

    def generate_embedding_index_from_configuration(self, configuration):
        first_layer = configuration[0]
        last_layer = configuration[-1]
        index = (self.layer_positions[first_layer][0], self.layer_positions[last_layer][1])
        return index

    def run_full_pipeline(self, train_images, train_labels, test_images, test_labels, layer_combinations='full',
                          fne='tf', verbose=False):
        # Ensure proper label encoding for sklearn
        train_labels, test_labels = encode_labels(train_labels, test_labels)

        initial_time = time()
        features_train = self.extract_features(train_images, fne)
        features_test = self.extract_features(test_images, fne)

        fne_features_train, l2_features_train = self.execute_postprocessings(features_train)
        fne_features_test, l2_features_test = self.execute_postprocessings(features_test)

        if fne == 'tf' or fne == 'keras':
            configurations = self.generate_configurations(layer_combinations)
            best_accuracy = 0
            for configuration in configurations:
                index = self.generate_embedding_index_from_configuration(configuration)
                prediction_fne = self.train_svm_with_features(fne_features_train[:, index[0]:index[1]], train_labels,
                                                              fne_features_test[:, index[0]:index[1]])
                accuracy_fne = accuracy(test_labels, prediction_fne)
                prediction_l2 = self.train_svm_with_features(l2_features_train[:, index[0]:index[1]], train_labels,
                                                             l2_features_test[:, index[0]:index[1]])
                accuracy_l2 = accuracy(test_labels, prediction_l2)
                best_accuracy = np.amax([best_accuracy, accuracy_fne, accuracy_l2])
                if verbose:
                    print('Configuration: {} \nAccuracy L2: {} \nAccuracy FNE: {}'.format(configuration, accuracy_l2,
                                                                                          accuracy_fne))
                    print(20 * '--')
        else:
            prediction_fne = self.train_svm_with_features(fne_features_train, train_labels, fne_features_test)
            accuracy_fne = accuracy(test_labels, prediction_fne)
            prediction_l2 = self.train_svm_with_features(l2_features_train, train_labels, l2_features_test)
            accuracy_l2 = accuracy(test_labels, prediction_l2)

            best_accuracy = np.amax([accuracy_fne, accuracy_l2])
            if verbose:
                print('Accuracy L2: {} \nAccuracy FNE: {}'.format(accuracy_l2, accuracy_fne))
                print(20 * '--')

        computation_time = time() - initial_time
        return best_accuracy, computation_time

    def run_single_pipeline(self, train_images, train_labels, test_images, test_labels, layer_combinations='full',
                            classifier_method=LinearSVC(), fne='tf', include_l2=True):
        if classifier_method == 'svm':
            classifier_method = LinearSVC()
        elif classifier_method == 'lda':
            classifier_method = LinearDiscriminantAnalysis()

        # Ensure proper label encoding for sklearn
        train_labels = np.repeat(train_labels, self.n_crops)
        test_labels = np.repeat(test_labels, self.n_crops)

        train_labels, test_labels = encode_labels(train_labels, test_labels)

        # Feature extraction
        initial_time = time()
        features_train = self.extract_features(train_images, fne)
        extract_train_time = time() - initial_time
        features_test = self.extract_features(test_images, fne)
        extract_test_time = time() - extract_train_time - initial_time

        # Feature postprocessing
        fne_features_train, l2_features_train = self.execute_postprocessings(features_train, include_l2)
        fne_features_test, l2_features_test = self.execute_postprocessings(features_test, include_l2)
        post_processings_time = time() - extract_test_time - extract_train_time - initial_time

        if fne == 'tf' or fne == 'keras':
            # Generate combinations of layers
            configurations = self.generate_configurations(layer_combinations)
            results = {'layer_combination': [], 'time': [], 'fne': [], 'l2': []}

            # Train SVMs and compute results
            for configuration in configurations:
                index = self.generate_embedding_index_from_configuration(configuration)

                start_fne_time = time()
                prediction_fne = self.train_with_features(classifier_method,
                                                          fne_features_train[:, index[0]:index[1]], train_labels,
                                                          fne_features_test[:, index[0]:index[1]])
                accuracy_fne = accuracy_average_per_class_voting(test_labels, prediction_fne, self.n_crops)
                fne_time = time() - start_fne_time

                results['layer_combination'].append(configuration)
                results['time'].append({'extract_train_time': extract_train_time,
                                        'extract_test_time': extract_test_time,
                                        'post_processings_time': post_processings_time, 'fne_time': fne_time})
                results['fne'].append(accuracy_fne)

                if include_l2:
                    prediction_l2 = self.train_with_features(classifier_method,
                                                             l2_features_train[:, index[0]:index[1]], train_labels,
                                                             l2_features_test[:, index[0]:index[1]])
                    accuracy_l2 = accuracy_average_per_class_voting(test_labels, prediction_l2, self.n_crops)
                    l2_time = time() - start_fne_time - fne_time
                    results['time'][0]['l2_time'] = l2_time
                    results['l2'].append(accuracy_l2)
        else:
            results = {'time': [], 'fne': [], 'l2': []}

            start_fne_time = time()
            if classifier_method == 'lda':
                prediction_fne = self.train_with_features(LinearDiscriminantAnalysis(),
                                                          fne_features_train, train_labels, fne_features_test)
            else:
                prediction_fne = self.train_with_features(LinearSVC(),
                                                          fne_features_train, train_labels, fne_features_test)

            accuracy_fne = accuracy_average_per_class_voting(test_labels, prediction_fne, self.n_crops)
            fne_time = time() - start_fne_time

            results['time'].append({'extract_train_time': extract_train_time,
                                    'extract_test_time': extract_test_time,
                                    'post_processings_time': post_processings_time, 'fne_time': fne_time})
            results['fne'].append(accuracy_fne)

            if include_l2:
                if classifier_method == 'lda':
                    prediction_l2 = self.train_with_features(LinearDiscriminantAnalysis(),
                                                             l2_features_train, train_labels, l2_features_test)
                else:
                    prediction_l2 = self.train_with_features(LinearSVC(),
                                                             l2_features_train, train_labels, l2_features_test)

                accuracy_l2 = accuracy_average_per_class_voting(test_labels, prediction_l2, self.n_crops)
                l2_time = time() - start_fne_time - fne_time
                results['time'][0]['l2_time'] = l2_time
                results['l2'].append(accuracy_l2)

        return results
