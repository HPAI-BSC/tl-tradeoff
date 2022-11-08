import os

import numpy as np
from sklearn.preprocessing import LabelEncoder


def model_resource(filename):
    current_dir = os.path.dirname(__file__)
    resource_dir = os.path.abspath(os.path.join(current_dir, '../resources/models'))
    return os.path.join(resource_dir, filename)


def resource(filename):
    current_dir = os.path.dirname(__file__)
    resource_dir = os.path.abspath(os.path.join(current_dir, '../resources'))
    return os.path.join(resource_dir, filename)


def image(filename):
    currentdir = os.path.dirname(__file__)
    resourcedir = os.path.abspath(os.path.join(currentdir, '../images'))
    return os.path.join(resourcedir, filename)


def is_layer(layer_names, op):
    for layer in layer_names:
        if op.split('/')[0] == layer:
            return True
    return False


def is_operation(op):
    operation_names = ['Relu', 'Softmax']
    for operation_name in operation_names:
        if op.split('/')[1] == operation_name:
            return True
    return False


def find_tensor_names(variable_names, operation_names):
    # Currently only works with Relu and Softmax operations
    layer_names = [layer.split('/')[0] for layer in variable_names if 'kernel' in layer]
    tensor_names = [op + ':0' for op in operation_names if is_layer(layer_names, op) and is_operation(op)]
    return tensor_names


def encode_labels(y_train, y_test):
    """Encode the labels in a format suitable for sklearn
    """
    le = LabelEncoder()
    le.fit(np.append(y_train, y_test))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test


def within_range(numbers, minimum=4, maximum=8):
    """
    Returns how many numbers lie within minimum and maximum in a given list of numbers
    """
    count = 0
    for n in numbers:
        if minimum <= n <= maximum:
            count = count + 1
    return count
