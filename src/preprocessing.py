import tensorflow as tf
from tensorflow import keras


def get_preprocessing_function(model: str = 'VGG16'):
    if model == 'VGG16':
        return keras.applications.vgg16.preprocess_input
    if model == 'ResNet50':
        return tf.keras.applications.resnet50.preprocess_input
    if model == 'ResNet152':
        return tf.keras.applications.resnet50.preprocess_input
    else:
        raise NotImplementedError
