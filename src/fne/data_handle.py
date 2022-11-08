import os
import shutil

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fne.crops_generator import get_crop_images_list_with_input_reshape, get_crop_images_list_with_min_axis_size
# from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from PIL import Image


def downaxis_shape(min_axis_size, height, width):
    if height < width:  # Landscape
        target_height = min_axis_size
        down_ratio = target_height / height
        target_width = round(width * down_ratio)
    else:  # Portrait (or squared)
        target_width = min_axis_size
        down_ratio = target_width / width
        target_height = round(height * down_ratio)

    return target_height, target_width


class DataHandle:
    def __init__(self):
        pass

    @staticmethod
    def generate_reduced_mnist():
        mnist_dir = 'data/mnist'
        mnist_dir_train = os.path.join(mnist_dir, 'train')
        mnist_dir_test = os.path.join(mnist_dir, 'test')
        new_train_path = 'resources/mnist/train'
        new_test_path = 'resources/mnist/test'
        os.makedirs(new_train_path, exist_ok=True)
        os.makedirs(new_test_path, exist_ok=True)
        for label_dir in os.listdir(mnist_dir_train):
            train_dir_path = os.path.join(mnist_dir_train, label_dir)
            os.makedirs(os.path.join(new_train_path, label_dir), exist_ok=True)
            for train_img in os.listdir(train_dir_path)[:100]:
                shutil.copy(os.path.join(train_dir_path, train_img), os.path.join(new_train_path, label_dir, train_img))
        for label_dir in os.listdir(mnist_dir_test):
            test_dir_path = os.path.join(mnist_dir_test, label_dir)
            os.makedirs(os.path.join(new_test_path, label_dir), exist_ok=True)
            for train_img in os.listdir(test_dir_path)[:60]:
                shutil.copy(os.path.join(test_dir_path, train_img), os.path.join(new_test_path, label_dir, train_img))
        print('Reduced mnist generated and saved on resources.')

    @staticmethod
    def _iterate_over_directories(data_dir, num_classes):
        image_paths = []
        labels = []
        for folder_class in sorted(os.listdir(data_dir)):
            if os.path.isdir(os.path.join(data_dir, folder_class)):
                folder_dir_path = os.path.join(data_dir, folder_class)
                if os.path.isdir(folder_dir_path):
                    for image in sorted(os.listdir(folder_dir_path)):
                        image_paths.append(os.path.join(folder_dir_path, image))
                        labels.append(folder_class)
                    num_classes -= 1
                    if num_classes == 0:
                        break
        return image_paths, labels

    def load_target_task_imgs_labels(self, data_dir, n_classes=-1, verbose=False):
        if verbose:
            print('Loading dataset', data_dir.split('/')[-1])
        train_data_dir = os.path.join(data_dir, 'train')
        validation_data_dir = os.path.join(data_dir, 'test')

        train_images, train_labels = self._iterate_over_directories(train_data_dir, n_classes)
        test_images, test_labels = self._iterate_over_directories(validation_data_dir, n_classes)
        if verbose:
            print('Total train images:', len(train_images), ' with their corresponding', len(train_labels), 'labels')
            print('Total test images:', len(test_images), ' with their corresponding', len(test_labels), 'labels')

        return train_images, train_labels, test_images, test_labels

    @staticmethod
    def check_the_channels_first(input_tensor_shape: [int, int, int, int]):
        """
        Returns true if the input tensor expects the images ordered with the 3 channels first: (N x 3 x H x W)
        """
        if input_tensor_shape[1] == 3:
            return True
        else:
            return False

    def load_image_array(self, img_path, input_reshape):
        pass

    def preprocess_image(self, abs_path, input_reshape, variable_input_shape, max_resolution, number_crops):
        if input_reshape is not None:
            # All images resized to input_reshape
            array_image = self.load_image_array(abs_path, input_reshape)
        elif variable_input_shape and number_crops == 1:
            # All images are updated to biggest image shape using padding
            img = Image.open(abs_path)
            img.load()
            array_image = self.padding(np.asarray(img), max_resolution)
        else:
            # All images have the same shape
            img = Image.open(abs_path)
            img.load()
            array_image = np.asarray(img)
        return array_image

    def load_image_and_resize(self, img_path, input_reshape, variable_input_shape, max_resolution, reduction_factor,
                              min_axis_size=None, channels_first=False, load_crops=False, number_of_crops=1):
        """
        Some models receive the input as (N x H x W x 3) and another ones as (N x 3 x H x W).
        So, depending on the model you need to change the image shape. That is what channels_first
        parameter controls.
        """

        abs_path = os.path.abspath(img_path)

        # img_path should have structure some_path/dataset/split/label/image.jpg
        # The crops should be stored in some_path/dataset_crops/split/label/image_crop_i.jpg or
        # some_path/dataset_crops/label/image_crop_i.jpg
        if number_of_crops == 10 and load_crops is True:
            folder, image = os.path.split(img_path)
            aux, label = os.path.split(folder)
            aux2, split = os.path.split(aux)
            path, dataset = os.path.split(aux2)

            crop_dir = os.path.join(path, dataset + '_crops', split, label, image)

            crop_list = []

            for i in range(number_of_crops):
                crop_name, crop_ext = os.path.splitext(crop_dir)
                crop_path = os.path.join(crop_dir, crop_name + '_crop_{}'.format(i) + crop_ext)

                array_image = self.preprocess_image(crop_path, input_reshape, variable_input_shape,
                                                    max_resolution, number_of_crops)
                crop_list.append(array_image)
            return crop_list

        elif number_of_crops == 10 and load_crops is False:
            crop_list = []
            if input_reshape is not None:
                crop_list = get_crop_images_list_with_input_reshape(abs_path, reduction_factor, input_reshape)
            elif reduction_factor is not None:
                crop_list = get_crop_images_list_with_min_axis_size(abs_path, reduction_factor, min_axis_size)
                if variable_input_shape:
                    for i, crop_array in enumerate(crop_list):
                        array_image = self.padding(crop_array, max_resolution)
                        crop_list[i] = array_image
            return crop_list

        else:

            array_image = self.preprocess_image(abs_path, input_reshape, variable_input_shape, max_resolution,
                                                number_of_crops)
            image = np.expand_dims(array_image, axis=0)

            if channels_first:
                image = np.rollaxis(image, 3, 1)

            return image

    def get_batch_image_array_shape(self, batch_images_path, input_reshape,
                                    number_of_crops, reduction_factor, min_axis_size=None, load_crops =True):
        variable_input_shape = False
        if load_crops:
            if input_reshape is not None:
                shape = input_reshape
            else:
                shape, variable_input_shape = self.get_max_resolution(batch_images_path,
                                                                      min_axis_size=min_axis_size)
        elif input_reshape is None:
            max_resolution, variable_input_shape = self.get_max_resolution(batch_images_path,
                                                                           min_axis_size=min_axis_size)
            shape = max_resolution
        else:
            shape = input_reshape

        if reduction_factor is not None and number_of_crops == 10:
            shape = (round(shape[0] * reduction_factor),
                     round(shape[1] * reduction_factor))
        return shape, variable_input_shape

    @staticmethod
    def data_generators(train_data_dir, validation_data_dir, input_reshape, batch_size):
        # Initiate the train and test generators with data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1. / 255)

        val_datagen = ImageDataGenerator(
            rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=input_reshape,
            batch_size=batch_size,
            class_mode="categorical")

        validation_generator = val_datagen.flow_from_directory(
            validation_data_dir,
            target_size=input_reshape,
            batch_size=batch_size,
            shuffle=False,
            class_mode="categorical")
        return train_generator, validation_generator

    @staticmethod
    def load_target_task(train_data_dir, validation_data_dir):
        # Get num classes in target task
        target_classes = len([name for name in os.listdir(train_data_dir)
                              if os.path.isdir(os.path.join(train_data_dir, name))])

        # Get num instances in training
        nb_train_samples = 0
        for root, dirs, files in os.walk(train_data_dir):
            nb_train_samples += len(files)

        # Get num instances in test
        nb_validation_samples = 0
        for root, dirs, files in os.walk(validation_data_dir):
            nb_validation_samples += len(files)

        return target_classes, train_data_dir, validation_data_dir, nb_train_samples, nb_validation_samples

    @staticmethod
    def preprocessing(image):

        max = np.amax(image)

        if max > 1:
            preprocessed_image = np.true_divide(image, max)
            return preprocessed_image

        else:
            return image

    def get_max_resolution(self, batch_images, min_axis_size=None):
        pass

    def padding(self, image_array, shape):
        image = np.pad(image_array, (*[((shape[i] - image_array.shape[i]) // 2,
                              ((shape[i] - image_array.shape[i]) // 2) + ((shape[i] - image_array.shape[i]) % 2)) for i in range(2)],
                           (0, 0)), mode='constant', constant_values=0.)
        return image
