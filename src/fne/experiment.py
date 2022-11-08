import csv
import os

from fne.data_handle import DataHandle
from fne.model import ModelHandle
from fne.full_pipeline import FullPipeline
from fne.pipeline import Pipeline


def file_is_image(file):
    if file.endswith('.png') or file.endswith('jpg') or file.endswith('jpeg'):
        return True
    return False

def read_imgs_csv(filename_csv):
    readed_imgs = []
    with open(filename_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            readed_imgs.append(row)
    return readed_imgs


def write_imgs_csv(imgs_list, filename_csv):
    with open(filename_csv, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        for img in imgs_list:
            csv_writer.writerow(img)


def get_experiments_configs(samples_per_class_list, n_splits, classifier_method):
    exp_configs = []
    for spc in samples_per_class_list:
        if spc == samples_per_class_list[-1]:
            i = 0
            exp_configs.append({'samples_per_class': spc, 'split': i, 'classifier': classifier_method})
        else:
            for i in range(n_splits):
                exp_configs.append({'samples_per_class': spc, 'split': i, 'classifier': classifier_method})
    return exp_configs


class Experiment:

    def __init__(self, config, config_exp, data_handle: DataHandle, model_handle: ModelHandle):
        self.train_filename = config['train_filename']
        self.test_filename = config['test_filename']
        self.train_images_path = config['train_images_path']
        self.test_images_path = config['test_images_path']
        self.splits_path = config['splits_path']
        self.n_crops = config['n_crops']
        self.load_crops = config['load_crops']
        self.reduction_factor = config['reduction_factor']
        try:
            self.min_axis_size = config['min_axis_size']
        except KeyError:
            self.min_axis_size = None

        self.config_exp = config_exp
        self.data_handle = data_handle
        self.model_handle = model_handle

        self.batch_size = config['batch_size']
        self.layer_combinations = config['layer_combinations']
        self.classifier = config['classifier_method']

        self.fne = config['fne']
        self.pipeline_type = config['pipeline']
        self.mode = config['mode']
        self.model_path = config['models_path']
        self.input_reshape = config['input_reshape']
        self.input_tensor = config['input_tensor']
        self.target_tensors = config['target_tensors']


    def get_experiment_paths(self):
        samples_per_class = self.config_exp['samples_per_class']
        split = self.config_exp['split']
        train_filename = os.path.join(self.splits_path,
                                      'train_samples_{:02d}_split_{:01d}.csv'.format(samples_per_class, split))
        test_filename = os.path.join(self.splits_path, self.test_filename)
        return train_filename, test_filename

    def build_imgs_labels_lists(self, imgs):
        images = []
        labels = []
        for img in imgs:
            images.append(img[0])
            labels.append(img[3])
        return images, labels

    def build_imgs_labels_lists_from_path(self, path):
        images = []
        labels = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file_is_image(file):
                    images.append(os.path.join(root, file))
                    labels.append(os.path.join(root, file).split('/')[-2])
        return images, labels

    def get_images_paths_and_labels(self):
        if self.train_images_path is not None and self.test_images_path is not None:
            train_images, train_labels = self.build_imgs_labels_lists_from_path(self.train_images_path)
            test_images, test_labels = self.build_imgs_labels_lists_from_path(self.test_images_path)
            return train_images, train_labels, test_images, test_labels

        if self.train_filename is None:
            self.train_filename, self.test_filename = self.get_experiment_paths()

        imgs_train = read_imgs_csv(self.train_filename)
        imgs_test = read_imgs_csv(self.test_filename)

        train_images, train_labels = self.build_imgs_labels_lists(imgs_train)
        test_images, test_labels = self.build_imgs_labels_lists(imgs_test)
        return train_images, train_labels, test_images, test_labels

    def build_pipeline(self):
        self.pipeline = Pipeline(model_handle=self.model_handle, data_handle=self.data_handle,
                                 batch_size=self.batch_size,
                                 n_crops=self.n_crops,
                                 load_crops=self.load_crops,
                                 reduction_factor=self.reduction_factor,
                                 min_axis_size=self.min_axis_size)

    def run_pipeline(self):
        train_images, train_labels, test_images, test_labels = self.get_images_paths_and_labels()
        accuracies = self.pipeline.run_single_pipeline(train_images, train_labels,
                                                       test_images, test_labels,
                                                       layer_combinations=self.layer_combinations,
                                                       classifier_method=self.classifier,
                                                       fne=self.fne)
        return accuracies

    def run_full_pipeline(self):
        train_images, train_labels, test_images, test_labels = self.get_images_paths_and_labels()
        best_accuracy, computation_time = self.full_pipeline.run_full_pipeline(train_images, train_labels,
                                                                               test_images, test_labels,
                                                                               self.layer_combinations, self.mode)
        return best_accuracy, computation_time

    def run_experiment(self):
        if self.pipeline_type == 'single':
            self.build_pipeline()
            accuracies = self.run_pipeline()
            return accuracies
        elif self.pipeline_type == 'full':
            self.build_full_pipeline()
            best_accuracy, computation_time = self.run_full_pipeline()
            return [best_accuracy, computation_time]

    # DEPRECATED

    def build_full_pipeline(self):
        self.full_pipeline = FullPipeline(model_path=self.model_path, batch_size=self.batch_size,
                                          image_resize=self.input_reshape, input_tensor=self.input_tensor,
                                          target_tensors=self.target_tensors, data_handle=self.data_handle,
                                          n_crops=self.n_crops, load_crops=self.load_crops,
                                          reduction_factor=self.reduction_factor)

    def run_single_pipeline(self):
        train_images, train_labels, test_images, test_labels = self.get_images_paths_and_labels()
        accuracies = self.full_pipeline.run_single_pipeline(train_images, train_labels,
                                                            test_images, test_labels,
                                                            layer_combinations=self.layer_combinations)
        return accuracies

