from time import time

import numpy as np
import itertools as it
import tensorflow as tf
import multiprocessing as mp
from sklearn.svm import LinearSVC
from tensorflow.python.keras.models import model_from_json

from hpai_utils.accuracy import accuracy_average_per_class, accuracy
from fne.data_handle import DataHandle
from fne.fne_keras import FeatureExtractionKeras
from fne.fne_tf import FeatureExtractionTensorflow
from fne.fne_torch import FeatureExtractionTorch
from fne.loaded_model import LoadedModelFromSession
from fne.post_processing import PostProcessing
from fne.utils import encode_labels


class FullPipeline:
    def __init__(self, model_path, batch_size, image_resize, input_tensor, target_tensors, data_handle: DataHandle,
                 torch_model=None, n_crops=1, load_crops=False, reduction_factor=None, min_axis_size=None):
        self.model_path = model_path
        self.batch_size = batch_size
        self.image_resize = image_resize
        self.input_tensor = input_tensor
        self.target_tensors = target_tensors
        self.layer_positions = {}
        self.data_handle = data_handle
        self.n_crops = n_crops
        self.load_crops = load_crops
        self.reduction_factor = reduction_factor
        self.min_axis_size = min_axis_size
        self.torch_model = torch_model

    def _load_model(self):
        if self.torch_model is not None:
            model = self.torch_model
        elif '.h5' in self.model_path:
            model = tf.keras.models.load_model(self.model_path, compile=False)
        elif '.hdf5' in self.model_path:
            json_path = self.model_path.replace('.hdf5', '.json')
            json_file = open(json_path, 'r')
            model_json = json_file.read()
            json_file.close()
            model = model_from_json(model_json)
            model.load_weights(self.model_path)
        else:
            raise NotImplementedError
        return model

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
        self.layer_positions = layer_positions
        return layer_positions

    def extract_features(self, images, fne='tf'):
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
            model = self._load_model()
            torch_fe = FeatureExtractionTorch(model, images, self.batch_size, self.target_tensors, self.image_resize,
                                              self.data_handle, self.n_crops, self.load_crops, self.reduction_factor,
                                              self.min_axis_size)
            features = torch_fe.extract_features()
        return features

    @staticmethod
    def train_svm_with_features(features_train, train_labels, features_test):
        clf = LinearSVC()
        clf.fit(X=features_train, y=train_labels)
        predicted_labels = clf.predict(features_test)
        return predicted_labels

    def generate_configurations(self, layer_combinations):
        max_size_subset = len(self.target_tensors) + 1
        combinations = []
        if layer_combinations == 'all':
            for k in range(1, len(self.target_tensors) + 1):
                for conf in list(it.combinations(self.target_tensors, k)):
                    combinations.append(list(conf))
        elif 'combinations_up_to_' in layer_combinations:
            for k in range(1, int(layer_combinations[19:]) + 1):
                for conf in list(it.combinations(self.target_tensors, k)):
                    combinations.append(list(conf))
        elif 'combinations_of_' in layer_combinations:
            for conf in list(it.combinations(self.target_tensors, int(layer_combinations[16:]))):
                combinations.append(list(conf))
        elif layer_combinations == 'all_contiguous':
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

    def execute_postprocessings(self, features):
        fne_features = PostProcessing().full_network_embedding(features)
        l2_features = PostProcessing().l2_norm(features, self.layer_positions)
        return fne_features, l2_features

    def generate_embedding_index_from_configuration(self, configuration):
        first_layer = configuration[0]
        last_layer = configuration[-1]
        index = (self.layer_positions[first_layer][0], self.layer_positions[last_layer][1])
        return index

    def run_single_experiment(self, configuration, fne_features_train, fne_features_test, l2_features_train,
                              l2_features_test, train_labels, test_labels, fne='tf', verbose=False):
        if fne == 'tf' or fne == 'keras':
            index = self.generate_embedding_index_from_configuration(configuration)
            # fne
            start_time = time()
            prediction_fne = self.train_svm_with_features(fne_features_train[:, index[0]:index[1]], train_labels,
                                                          fne_features_test[:, index[0]:index[1]])
            accuracy_fne = accuracy(test_labels, prediction_fne)
            fne_time = time() - start_time
            # l2
            prediction_l2 = self.train_svm_with_features(l2_features_train[:, index[0]:index[1]], train_labels,
                                                         l2_features_test[:, index[0]:index[1]])
            accuracy_l2 = accuracy(test_labels, prediction_l2)
            l2_time = time() - fne_time - start_time
            # collect result
            result = {'configuration': configuration,
                      'accuracy': {'fne': accuracy_fne, 'l2': accuracy_l2},
                      'time': {'fne': fne_time, 'l2': l2_time}}
        else:
            # fne
            start_time = time()
            prediction_fne = self.train_svm_with_features(fne_features_train, train_labels, fne_features_test)
            accuracy_fne = accuracy(test_labels, prediction_fne)
            fne_time = time() - start_time

            # l2
            prediction_l2 = self.train_svm_with_features(l2_features_train, train_labels, l2_features_test)
            accuracy_l2 = accuracy(test_labels, prediction_l2)
            l2_time = time() - fne_time - start_time
            # collect result
            result = {'accuracy': {'fne': accuracy_fne, 'l2': accuracy_l2},
                      'time': {'fne': fne_time, 'l2': l2_time}}
        if verbose:
            print(result)
            print(20 * '--')

        return result

    def run_full_pipeline(self, train_images, train_labels, test_images, test_labels, layer_combinations='full',
                          mode='seq', fne='tf', verbose=False):
        # Ensure proper label encoding for sklearn
        train_labels, test_labels = encode_labels(train_labels, test_labels)

        # Feature extraction
        initial_time = time()
        features_train = self.extract_features(train_images, fne)
        features_test = self.extract_features(test_images, fne)

        # Feature postprocessing
        fne_features_train, l2_features_train = self.execute_postprocessings(features_train)
        fne_features_test, l2_features_test = self.execute_postprocessings(features_test)

        if fne == 'tf' or fne == 'keras':
            # Generate combinations of layers
            configurations = self.generate_configurations(layer_combinations)
        results = []
        # best_configuration = None
        best_accuracy = 0

        if mode == 'seq':
            if fne == 'tf' or fne == 'keras':
                for configuration in configurations:
                    results.append(self.run_single_experiment(configuration, fne_features_train, fne_features_test,
                                                              l2_features_train, l2_features_test, train_labels,
                                                              test_labels, verbose))
            else:
                configuration = None
                results.append(self.run_single_experiment(configuration, fne_features_train, fne_features_test,
                                                          l2_features_train, l2_features_test, train_labels,
                                                          test_labels, verbose))

        elif mode == 'sync':
            if fne == 'tf' or fne == 'keras':
                with mp.Pool(mp.cpu_count()) as pool:
                    args = [(config, fne_features_train, fne_features_test, l2_features_train,
                             l2_features_test, train_labels, test_labels, verbose) for config in configurations]
                    results = pool.starmap(self.run_single_experiment, args)
            else:
                config = None
                with mp.Pool(mp.cpu_count()) as pool:
                    args = [(config, fne_features_train, fne_features_test, l2_features_train,
                                 l2_features_test, train_labels, test_labels, verbose)]
                    results = pool.starmap(self.run_single_experiment, args)
        elif mode == 'async':
            if fne == 'tf' or fne == 'keras':
                with mp.Pool(mp.cpu_count()) as pool:
                    args = [(config, fne_features_train, fne_features_test, l2_features_train,
                             l2_features_test, train_labels, test_labels, verbose) for config in configurations]
                    results = pool.starmap_async(self.run_single_experiment, args).get()
            else:
                config = None
                with mp.Pool(mp.cpu_count()) as pool:
                    args = [(config, fne_features_train, fne_features_test, l2_features_train,
                             l2_features_test, train_labels, test_labels, verbose)]
                    results = pool.starmap_async(self.run_single_experiment, args).get()

        else:
            raise Exception('No valid mode provided. Valid values are seq, sync and async')

        for result in results:
            accuracies = result['accuracy']
            accuracy = np.max([best_accuracy, accuracies['fne'], accuracies['l2']])
            if accuracy > best_accuracy:
                # best_configuration = result['configuration']
                best_accuracy = accuracy

        computation_time = time() - initial_time
        return best_accuracy, computation_time

    def run_single_pipeline(self, train_images, train_labels, test_images, test_labels,
                            layer_combinations='full', fne='tf'):
        # Ensure proper label encoding for sklearn
        train_labels, test_labels = encode_labels(train_labels, test_labels)

        # Feature extraction
        initial_time = time()
        features_train = self.extract_features(train_images, fne)
        extract_train_time = time() - initial_time
        features_test = self.extract_features(test_images, fne)
        extract_test_time = time() - extract_train_time - initial_time

        # Feature postprocessing
        fne_features_train, l2_features_train = self.execute_postprocessings(features_train)
        fne_features_test, l2_features_test = self.execute_postprocessings(features_test)
        post_processings_time = time() - extract_test_time - extract_train_time - initial_time

        if fne == 'tf' or fne == 'keras':
            # Generate combinations of layers
            configurations = self.generate_configurations(layer_combinations)
            results = {'layer_combination': [], 'time': [], 'fne': [], 'l2': []}

            # Train SVMs and compute results
            for configuration in configurations:
                index = self.generate_embedding_index_from_configuration(configuration)
                start_fne_time = time()
                prediction_fne = self.train_svm_with_features(fne_features_train[:, index[0]:index[1]], train_labels,
                                                              fne_features_test[:, index[0]:index[1]])
                accuracy_fne = accuracy_average_per_class(test_labels, prediction_fne)
                fne_time = time() - start_fne_time
                prediction_l2 = self.train_svm_with_features(l2_features_train[:, index[0]:index[1]], train_labels,
                                                             l2_features_test[:, index[0]:index[1]])
                accuracy_l2 = accuracy_average_per_class(test_labels, prediction_l2)
                l2_time = time() - start_fne_time - fne_time

                results['layer_combination'].append(configuration)
                results['time'].append({'extract_train_time': extract_train_time, 'extract_test_time': extract_test_time,
                                        'post_processings_time': post_processings_time, 'fne_time': fne_time,
                                        'l2_time': l2_time})
                results['fne'].append(accuracy_fne)
                results['l2'].append(accuracy_l2)
        else:
            results = {'time': [], 'fne': [], 'l2': []}
            start_fne_time = time()
            prediction_fne = self.train_svm_with_features(fne_features_train, train_labels, fne_features_test)
            accuracy_fne = accuracy_average_per_class(test_labels, prediction_fne)
            fne_time = time() - start_fne_time
            prediction_l2 = self.train_svm_with_features(l2_features_train, train_labels, l2_features_test)
            accuracy_l2 = accuracy_average_per_class(test_labels, prediction_l2)
            l2_time = time() - start_fne_time - fne_time
            results['time'].append({'extract_train_time': extract_train_time, 'extract_test_time': extract_test_time,
                                    'post_processings_time': post_processings_time, 'fne_time': fne_time,
                                    'l2_time': l2_time})
            results['fne'].append(accuracy_fne)
            results['l2'].append(accuracy_l2)

        return results
