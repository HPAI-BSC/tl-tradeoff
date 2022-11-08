import csv
import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras


def get_img_paths_and_labels(split_path):
    with open(split_path) as file:
        c = csv.reader(file)
        aux_list = [(x[0], x[3]) for x in c]
        img_list = [x[0] for x in aux_list]
        label_list = [x[1] for x in aux_list]

    return img_list, label_list


def get_dataset(split_path, crop_dataset_path, n_crops):
    with open(split_path) as sfile:
        c = csv.reader(sfile)
        aux_list = [(x[0], x[3]) for x in c]
        img_list = [x[0] for x in aux_list]
        label_list = [x[1] for x in aux_list]

        label_list_ordered = list(set(label_list))
        label_list_ordered.sort()
        label_dict = {s: i for i, s in enumerate(label_list_ordered)}
        label_list = [str(label_dict[s]) for s in label_list]

    for i, img in enumerate(img_list):
        # Adds '_crops' to the dataset name and the '_crop_' structure to the filename
        path, ext = os.path.splitext(img)
        path2, filename = os.path.split(path)
        path3, label = os.path.split(path2)
        path4, split = os.path.split(path3)
        path5, dataset = os.path.split(path4)
        crops = [os.path.join(crop_dataset_path, split, label, filename + '_crop_' + str(i) + ext)
                 for i in range(n_crops)] if n_crops > 1 else [
            os.path.join(path5, dataset, split, label, filename + ext)]
        img_list[i] = crops
    img_list = [x for y in img_list for x in y]
    label_list = [x for y in label_list for x in [y] * n_crops]

    return tf.data.Dataset.from_tensor_slices(list(zip(img_list, label_list)))


def get_optimizer(optimizer, learning_rate, momentum, weight_decay):
    if optimizer == 'sgd':
        return tfa.optimizers.SGDW(learning_rate=learning_rate, momentum=momentum,
                                   weight_decay=weight_decay, clipnorm=1.0)
    elif optimizer == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    else:
        raise NotImplementedError


def load_image(path):
    return np.array(keras.utils.load_img(path.numpy().decode('utf-8')))


def preprocess_batch(batch, preprocess_function):
    images = batch[:, 0]
    labels = batch[:, 1]
    return preprocess_function(tf.map_fn(load_image, images, fn_output_signature=tf.int32)), \
           tf.map_fn(lambda n: int(n.numpy()), labels, fn_output_signature=tf.float32)


def get_experiments_configs(samples_per_class_list, n_splits):
    exp_configs = []
    for spc in samples_per_class_list:
        if spc == samples_per_class_list[-1]:
            i = 0
            exp_configs.append({'samples_per_class': spc, 'split': i})
        else:
            for i in range(n_splits):
                exp_configs.append({'samples_per_class': spc, 'split': i})
    return exp_configs
